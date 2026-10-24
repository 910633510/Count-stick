import Testing
import UIKit
@testable import Count_stick

struct AutomaticRegionHistoryTests {
    @Test func legacyRecordsWithoutRegionStillDecode() throws {
        let id = UUID()
        let data = try JSONSerialization.data(withJSONObject: [[
            "id": id.uuidString, "date": 1000, "markers": [], "automaticCount": 0, "note": "Older saved count"
        ]])
        let records = try JSONDecoder().decode([CountRecord].self, from: data)
        #expect(records.count == 1)
        #expect(records[0].id == id)
        #expect(records[0].automaticRegion == nil)
        #expect(records[0].note == "Older saved count")
    }

    @MainActor @Test func regionSurvivesHistoryAndNeverRestrictsManualCorrection() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let marker = CountMarker(x: 0.5, y: 0.5)
        let region = AutoCountRegionResult(
            keptIDs: [marker.id], excluded: [],
            regions: [AutoCountBox(x: 0.3, y: 0.3, width: 0.4, height: 0.4)], fallbackReason: .none
        )
        let record = CountRecord(id: UUID(), date: Date(timeIntervalSince1970: 1000),
                                 markers: [marker], automaticCount: 1, note: "Checked",
                                 automaticRegion: region)
        let photo = UIGraphicsImageRenderer(size: CGSize(width: 100, height: 100)).image { context in
            UIColor.white.setFill()
            context.fill(CGRect(x: 0, y: 0, width: 100, height: 100))
        }
        try CountHistory(directory: directory).save(record, image: photo)
        let reopened = CountHistory(directory: directory)
        let restored = try #require(reopened.records.first)
        #expect(restored == record)
        let workspace = CountWorkspace()
        workspace.restore(restored, image: try #require(reopened.image(for: restored)))
        #expect(workspace.automaticRegion == region)
        #expect(workspace.isSaved)

        // A missed edge endpoint can be added anywhere in the photo, including outside the automatic range.
        workspace.add(at: CGPoint(x: 0.95, y: 0.05))
        #expect(workspace.edits.markers.count == 2)
        #expect(workspace.edits.markers.last?.isManual == true)
        #expect(!workspace.isSaved)
        #expect(workspace.automaticRegion == region)
        workspace.undo()
        #expect(workspace.edits.markers == [marker])
        #expect(workspace.isSaved)

        workspace.countManually()
        #expect(workspace.automaticRegion == nil)
        #expect(workspace.automaticCount == nil)
        workspace.restore(restored, image: photo)
        workspace.reset()
        #expect(workspace.automaticRegion == nil)
        #expect(workspace.image == nil)
    }
}
