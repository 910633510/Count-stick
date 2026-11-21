import SwiftUI
import UIKit

@MainActor
final class CountWorkspace: ObservableObject {
    enum Phase { case empty, processing, ready, failed }
    @Published private(set) var category: CountCategory = .skewers
    @Published private(set) var image: UIImage?
    @Published private(set) var edits = MarkerEdits()
    @Published private(set) var phase: Phase = .empty
    @Published private(set) var errorMessage: String?
    @Published private(set) var automaticCount: Int?
    @Published private(set) var automaticRegion: AutoCountRegionResult?
    @Published var note = ""
    @Published private(set) var saved = false
    @Published var subscriptionRequired = false
    @Published var accessError: String?
    private(set) var id = UUID()
    private(set) var date = Date()
    private var generation = UUID()
    private let access: CountingAccess
    private let router: CountDetectorRouter
    private let requestReservation: () async throws -> CountingAccess.Reservation
    private var reservingID: UUID?
    private var activeReservation: CountingAccess.Reservation?
    private var savedMarkers: [CountMarker]?
    private var savedNote: String?

    init(access: CountingAccess? = nil, detector: ((UIImage) async throws -> StickDetectionResult)? = nil,
         router: CountDetectorRouter? = nil,
         reservationProvider: (() async throws -> CountingAccess.Reservation)? = nil) {
        let access = access ?? .shared
        self.access = access
        self.requestReservation = reservationProvider ?? { try await access.reserveCount() }
        self.router = router ?? CountDetectorRouter(skewers: detector)
    }

    private func reserve() async -> CountingAccess.Reservation? {
        guard reservingID == nil, phase != .processing else { return nil }
        let requestID = UUID(), originalGeneration = generation, originalCategory = category
        reservingID = requestID
        defer { if reservingID == requestID { reservingID = nil } }
        do {
            let reservation = try await requestReservation()
            guard generation == originalGeneration, category == originalCategory,
                  reservingID == requestID, !Task.isCancelled else {
                access.cancelCount(reservation)
                return nil
            }
            return reservation
        } catch {
            guard generation == originalGeneration, category == originalCategory, reservingID == requestID else { return nil }
            if case CountAccessError.subscriptionRequired = error { subscriptionRequired = true }
            else { accessError = error.localizedDescription }
            return nil
        }
    }

    private func release(_ reservation: CountingAccess.Reservation) {
        access.cancelCount(reservation)
        if activeReservation?.id == reservation.id { activeReservation = nil }
    }

    /// The view confirms replacement of an unsaved result before invoking this method.
    func selectCategory(_ category: CountCategory) {
        guard category != self.category else { return }
        reset()
        self.category = category
    }

    var hasUnsavedChanges: Bool { image != nil && (savedMarkers != edits.markers || savedNote != note) }
    var isSaved: Bool { saved && savedNote == note && savedMarkers == edits.markers }
    var record: CountRecord { CountRecord(id: id, date: date, markers: edits.markers, automaticCount: automaticCount, note: note, automaticRegion: automaticRegion, category: category) }

    func load(data: Data) async {
        guard let reservation = await reserve() else { return }
        defer { release(reservation) }
        let token = beginLoading()
        activeReservation = reservation
        do {
            let image = try await Task.detached(priority: .userInitiated) { try PhotoProcessing.load(data: data) }.value
            guard generation == token else { return }
            self.image = image
            await detect(image, token: token, reservation: reservation)
        } catch {
            guard generation == token else { return }
            phase = .empty
            errorMessage = error.localizedDescription
        }
    }

    func load(image: UIImage) async {
        guard let reservation = await reserve() else { return }
        defer { release(reservation) }
        let token = beginLoading()
        activeReservation = reservation
        do {
            let prepared = try await Task.detached(priority: .userInitiated) { try PhotoProcessing.normalized(image) }.value
            guard generation == token else { return }
            self.image = prepared
            await detect(prepared, token: token, reservation: reservation)
        } catch {
            guard generation == token else { return }
            phase = .empty
            errorMessage = error.localizedDescription
        }
    }

    func reportImportError() {
        errorMessage = L10n.text("error.image")
    }

    func retry() async {
        guard let image, let reservation = await reserve() else { return }
        defer { release(reservation) }
        generation = UUID()
        activeReservation = reservation
        phase = .processing
        errorMessage = nil
        await detect(image, token: generation, reservation: reservation)
    }

    func countManually() {
        guard image != nil else { return }
        if let activeReservation { release(activeReservation) }
        generation = UUID()
        edits.replace(with: [])
        automaticCount = nil
        automaticRegion = nil
        phase = .ready
        errorMessage = nil
    }

    func add(at point: CGPoint) { edits.add(at: point) }
    func remove(id: UUID) { edits.remove(id: id) }
    func undo() { edits.undo() }
    func markSaved() { savedMarkers = edits.markers; savedNote = note; saved = true }
    func clearError() { errorMessage = nil }
    func savedRecordWasDeleted() { saved = false; savedMarkers = nil; savedNote = nil }

    func restore(_ record: CountRecord, image: UIImage) {
        reset()
        self.category = record.category
        self.id = record.id
        self.date = record.date
        self.image = image
        self.edits.replace(with: record.markers)
        self.automaticCount = record.automaticCount
        self.automaticRegion = record.automaticRegion
        self.note = record.note
        self.phase = .ready
        markSaved()
    }

    func reset() {
        if let activeReservation { release(activeReservation) }
        reservingID = nil
        generation = UUID()
        id = UUID()
        date = Date()
        image = nil
        edits.replace(with: [])
        phase = .empty
        automaticCount = nil
        automaticRegion = nil
        errorMessage = nil
        subscriptionRequired = false
        accessError = nil
        note = ""
        saved = false
        savedMarkers = nil
        savedNote = nil
    }

    private func beginLoading() -> UUID {
        reset()
        phase = .processing
        return generation
    }

    private func detect(_ image: UIImage, token: UUID, reservation: CountingAccess.Reservation) async {
        do {
            let result = try await router.detect(image, category: category)
            guard generation == token else { return }
            try Task.checkCancellation()
            try access.completeCount(reservation)
            edits.replace(with: result.markers)
            automaticCount = result.markers.count
            automaticRegion = result.automaticRegion
            phase = .ready
        } catch {
            guard generation == token else { return }
            errorMessage = error.localizedDescription
            phase = .failed
        }
    }
}
