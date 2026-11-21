import XCTest
import UIKit
import CryptoKit
@testable import Count_stick

@MainActor
final class PipeRealModelTests: XCTestCase {
    private var repository: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent() }

    func testQualifiedBundledPipeModelMatchesBothActualAppPhotoPaths() async throws {
        let manifest = try fixtures()
        let detector = PipeDetector()
        for row in manifest.images {
            let data = try source(row)
            let paths = [("gallery", try PhotoProcessing.load(data: data)),
                         ("camera", try PhotoProcessing.normalized(try XCTUnwrap(UIImage(data: data))))]
            for (variant, photo) in paths {
                let expected = try XCTUnwrap(row.expectations.first { $0.variant == variant })
                let result = try await detector.detect(from: photo)
                try compare(result.markers, expected: expected)
                XCTAssertEqual(result.markers, result.rawMarkers, "Pipe ROI must preserve every retained end")
                XCTAssertTrue(result.automaticRegion.excluded.isEmpty)
                XCTAssertEqual(Set(result.automaticRegion.keptIDs), Set(result.markers.map(\.id)))
            }
        }
    }

    func testRealPipeWorkspaceSuccessDebitsOnceAndDecodeFailureDoesNotDebit() async throws {
        let manifest = try fixtures()
        let row = try XCTUnwrap(manifest.images.first { $0.expectations.contains { $0.count > 0 } })
        let storage = TestCountUsage()
        let workspace = CountWorkspace(access: CountingAccess(storage: storage, usesStoreKit: false))
        workspace.selectCategory(.steelPipes)
        await workspace.load(data: try source(row))
        XCTAssertEqual(workspace.phase, .ready)
        XCTAssertEqual(workspace.automaticCount, row.expectations.first { $0.variant == "gallery" }?.count)
        XCTAssertEqual(storage.value, 1)
        XCTAssertEqual(workspace.record.category, .steelPipes)
        await workspace.load(data: Data([0, 1, 2]))
        XCTAssertEqual(storage.value, 1)
        XCTAssertNil(workspace.automaticCount)
    }

    private func compare(_ markers: [CountMarker], expected: Expected) throws {
        XCTAssertEqual(markers.count, expected.count)
        XCTAssertEqual(expected.centers.count, expected.count)
        var remaining = expected.centers
        for marker in markers {
            guard let nearest = remaining.indices.min(by: {
                hypot(remaining[$0][0] - marker.x, remaining[$0][1] - marker.y) <
                hypot(remaining[$1][0] - marker.x, remaining[$1][1] - marker.y)
            }) else { XCTFail("Unexpected extra marker"); return }
            let match = remaining.remove(at: nearest)
            XCTAssertLessThanOrEqual(abs(match[0] - marker.x), 0.00025)
            XCTAssertLessThanOrEqual(abs(match[1] - marker.y), 0.00025)
        }
        XCTAssertTrue(remaining.isEmpty)
    }

    private func fixtures() throws -> Fixtures {
        let path = repository.appendingPathComponent("output/pipes-20261026/app/real-model-test-fixtures.json")
        guard FileManager.default.fileExists(atPath: path.path),
              let configURL = Bundle.main.url(forResource: "PipeEnds", withExtension: "json"),
              Bundle.main.url(forResource: "PipeEnds", withExtension: "mlmodelc") != nil else {
            throw XCTSkip("Qualified model, final configuration and approved fixture manifest are not bundled yet.")
        }
        let manifest = try JSONDecoder().decode(Fixtures.self, from: Data(contentsOf: path))
        let config = try JSONDecoder().decode(PipeModelConfiguration.self, from: Data(contentsOf: configURL))
        try config.validate()
        XCTAssertEqual(manifest.status, "QUALIFIED_PIPE_MODEL_E2E_FIXTURES")
        XCTAssertEqual(manifest.modelSHA256, config.checkpointSHA256)
        XCTAssertEqual(manifest.exportID, config.exportID)
        XCTAssertFalse(manifest.testContentIncluded)
        XCTAssertFalse(manifest.images.isEmpty)
        return manifest
    }

    private func source(_ row: Fixture) throws -> Data {
        XCTAssertEqual(row.split, "val")
        let path = repository.appendingPathComponent(row.imagePath).standardizedFileURL
        XCTAssertTrue(path.path.hasPrefix(repository.path + "/output/pipes-20261026/"))
        let data = try Data(contentsOf: path)
        XCTAssertEqual(SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined(), row.sha256)
        return data
    }
    private struct Fixtures: Decodable {
        let status: String
        let modelSHA256: String
        let exportID: UUID
        let testContentIncluded: Bool
        let images: [Fixture]
    }
    private struct Fixture: Decodable {
        let id: String
        let split: String
        let imagePath: String
        let sha256: String
        let expectations: [Expected]
    }
    private struct Expected: Decodable {
        let variant: String
        let count: Int
        let centers: [[Double]]
    }
}
