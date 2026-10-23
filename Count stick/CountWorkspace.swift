import SwiftUI
import UIKit

@MainActor
final class CountWorkspace: ObservableObject {
    enum Phase { case empty, processing, ready, failed }
    @Published private(set) var image: UIImage?
    @Published private(set) var edits = MarkerEdits()
    @Published private(set) var phase: Phase = .empty
    @Published private(set) var errorMessage: String?
    @Published private(set) var automaticCount: Int?
    @Published private(set) var automaticRegion: AutoCountRegionResult?
    @Published var note = ""
    @Published private(set) var saved = false
    private(set) var id = UUID()
    private(set) var date = Date()
    private var generation = UUID()
    private let detector = StickDetector()
    private var savedMarkers: [CountMarker]?
    private var savedNote: String?

    var hasUnsavedChanges: Bool { image != nil && (savedMarkers != edits.markers || savedNote != note) }
    var isSaved: Bool { saved && savedNote == note && savedMarkers == edits.markers }
    var record: CountRecord { CountRecord(id: id, date: date, markers: edits.markers, automaticCount: automaticCount, note: note, automaticRegion: automaticRegion) }

    func load(data: Data) async {
        let token = beginLoading()
        do {
            let image = try await Task.detached(priority: .userInitiated) { try PhotoProcessing.load(data: data) }.value
            guard generation == token else { return }
            self.image = image
            await detect(image, token: token)
        } catch {
            guard generation == token else { return }
            phase = .empty
            errorMessage = error.localizedDescription
        }
    }

    func load(image: UIImage) async {
        let token = beginLoading()
        do {
            let prepared = try await Task.detached(priority: .userInitiated) { try PhotoProcessing.normalized(image) }.value
            guard generation == token else { return }
            self.image = prepared
            await detect(prepared, token: token)
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
        guard let image else { return }
        generation = UUID()
        phase = .processing
        errorMessage = nil
        await detect(image, token: generation)
    }

    func countManually() {
        guard image != nil else { return }
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
        generation = UUID()
        id = UUID()
        date = Date()
        image = nil
        edits.replace(with: [])
        phase = .empty
        automaticCount = nil
        automaticRegion = nil
        errorMessage = nil
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

    private func detect(_ image: UIImage, token: UUID) async {
        do {
            let result = try await detector.detectWithAutomaticRegion(from: image)
            guard generation == token else { return }
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
