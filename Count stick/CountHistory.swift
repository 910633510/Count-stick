import Foundation
import UIKit
import SwiftUI

struct CountRecord: Identifiable, Codable, Equatable {
    var id: UUID
    var date: Date
    var markers: [CountMarker]
    var automaticCount: Int?
    var note: String
    var count: Int { markers.count }
}

@MainActor
final class CountHistory: ObservableObject {
    @Published private(set) var records: [CountRecord] = []
    @Published var errorMessage: String?
    private let directory: URL
    private var loadFailed = false
    private let fileManager: FileManager

    init(directory: URL? = nil, fileManager: FileManager = .default) {
        self.fileManager = fileManager
        self.directory = directory ?? fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("CountHistory", isDirectory: true)
        do {
            try fileManager.createDirectory(at: self.directory, withIntermediateDirectories: true)
            var location = self.directory
            var values = URLResourceValues()
            values.isExcludedFromBackup = true
            try location.setResourceValues(values)
            let manifest = self.directory.appendingPathComponent("records.json")
            if fileManager.fileExists(atPath: manifest.path) {
                records = try JSONDecoder().decode([CountRecord].self, from: Data(contentsOf: manifest))
                // A missing image must remain visible as an error, not silently erase a record.
            }
        } catch { loadFailed = true; errorMessage = L10n.text("error.history.load") }
    }

    func image(for record: CountRecord, thumbnail: Bool = false) -> UIImage? {
        UIImage(contentsOfFile: imageURL(id: record.id, thumbnail: thumbnail).path)
    }

    func save(_ record: CountRecord, image: UIImage) throws {
        guard !loadFailed else { throw CocoaError(.fileReadCorruptFile) }
        guard let data = image.jpegData(compressionQuality: 0.9),
              let smallData = try PhotoProcessing.normalized(image, maximumDimension: 320).jpegData(compressionQuality: 0.8) else {
            throw CountError.unreadableImage
        }
        // Photos are immutable within a session; corrections only change its manifest entry.
        let fullURL = imageURL(id: record.id)
        let thumbURL = imageURL(id: record.id, thumbnail: true)
        let existed = fileManager.fileExists(atPath: fullURL.path)
        do {
            if !existed {
                try data.write(to: fullURL, options: [.atomic, .completeFileProtection])
                try smallData.write(to: thumbURL, options: [.atomic, .completeFileProtection])
            }
            var updated = records.filter { $0.id != record.id }
            updated.insert(record, at: 0)
            try persist(updated)
            records = updated
        } catch {
            if !existed {
                try? fileManager.removeItem(at: fullURL)
                try? fileManager.removeItem(at: thumbURL)
            }
            throw error
        }
    }

    func delete(_ record: CountRecord) throws {
        guard !loadFailed else { throw CocoaError(.fileReadCorruptFile) }
        // Retain the record until every file is removed, so a failed deletion can be retried.
        for url in [imageURL(id: record.id), imageURL(id: record.id, thumbnail: true)] {
            if fileManager.fileExists(atPath: url.path) { try fileManager.removeItem(at: url) }
        }
        let updated = records.filter { $0.id != record.id }
        try persist(updated)
        records = updated
    }

    private func persist(_ records: [CountRecord]) throws {
        try JSONEncoder().encode(records).write(to: directory.appendingPathComponent("records.json"), options: [.atomic, .completeFileProtection])
    }

    private func imageURL(id: UUID, thumbnail: Bool = false) -> URL {
        directory.appendingPathComponent(id.uuidString + (thumbnail ? "-thumb.jpg" : ".jpg"))
    }
}
