import Foundation

/// All boxes use upright image coordinates normalized to0...1, with top-left origin.
struct AutoCountBox: Codable, Hashable, Sendable {
    var x: Double
    var y: Double
    var width: Double
    var height: Double
    var centerX: Double { x + width / 2 }
    var centerY: Double { y + height / 2 }
    var area: Double { width * height }
    var isValid: Bool { [x, y, width, height].allSatisfy(\.isFinite) && width > 0 && height > 0 }

    func contains(x px: Double, y py: Double) -> Bool {
        px >= x && px <= x + width && py >= y && py <= y + height
    }
    func intersectionArea(with other: AutoCountBox) -> Double {
        max(0, min(x + width, other.x + other.width) - max(x, other.x)) *
        max(0, min(y + height, other.y + other.height) - max(y, other.y))
    }
    func union(_ other: AutoCountBox) -> AutoCountBox {
        let left = min(x, other.x), top = min(y, other.y)
        return AutoCountBox(x: left, y: top, width: max(x + width, other.x + other.width) - left,
                            height: max(y + height, other.y + other.height) - top)
    }
    func expanded(horizontal: Double, vertical: Double) -> AutoCountBox {
        AutoCountBox(x: x - horizontal, y: y - vertical, width: width + horizontal * 2, height: height + vertical * 2)
    }
    var clippedToImage: AutoCountBox {
        let left = max(0, x), top = max(0, y)
        return AutoCountBox(x: left, y: top, width: max(0, min(1, x + width) - left), height: max(0, min(1, y + height) - top))
    }
}

struct AutoCountDetection: Codable, Sendable {
    let id: UUID
    let box: AutoCountBox
    let confidence: Double
}

/// Contains no recognized string, so OCR content need not be persisted with history.
struct AutoCountTextArea: Codable, Sendable {
    let box: AutoCountBox
    let confidence: Double
    let characterCount: Int
}

struct AutoCountRegionResult: Codable, Equatable, Sendable {
    enum FallbackReason: String, Codable, Sendable { case none, sparse, ambiguous, empty }
    enum ExclusionReason: String, Codable, Sendable { case text, isolated }
    struct Exclusion: Codable, Equatable, Sendable {
        let id: UUID
        let reason: ExclusionReason
    }
    let keptIDs: [UUID]
    let excluded: [Exclusion]
    let regions: [AutoCountBox]
    let fallbackReason: FallbackReason
}

enum AutoCountRegion {
    /// Conservative grouping only filters existing full-image detections. It never crops,
    /// reruns the model, raises its score threshold, or deletes a second supported bundle.
    static func select(detections: [AutoCountDetection], textAreas: [AutoCountTextArea], imageAspectRatio: Double) -> AutoCountRegionResult {
        select(detections: detections, textAreas: textAreas, imageAspectRatio: imageAspectRatio, useText: true, useGeometry: true)
    }

    /// Separate switches support the predeclared development ablation; production uses select above.
    static func select(detections: [AutoCountDetection], textAreas: [AutoCountTextArea], imageAspectRatio: Double,
                       useText: Bool, useGeometry: Bool) -> AutoCountRegionResult {
        guard !detections.isEmpty else { return result([], [], [], .empty) }
        guard imageAspectRatio.isFinite, imageAspectRatio > 0,
              detections.allSatisfy({ $0.box.isValid && $0.box.clippedToImage.isValid &&
                  (0...1).contains($0.box.centerX) && (0...1).contains($0.box.centerY) &&
                  $0.box.width <= 1 && $0.box.height <= 1 && $0.confidence.isFinite }),
              Set(detections.map(\.id)).count == detections.count else {
            return result(detections.map(\.id), [], [], .ambiguous)
        }
        let aspect = imageAspectRatio
        let sizes = detections.map { sqrt($0.box.width * aspect * $0.box.height) }.sorted()
        let side = sizes[sizes.count / 2]
        guard side.isFinite, side > 0 else { return result(detections.map(\.id), [], [], .ambiguous) }
        func distance(_ a: AutoCountDetection, _ b: AutoCountDetection) -> Double {
            hypot((a.box.centerX - b.box.centerX) * aspect, a.box.centerY - b.box.centerY)
        }
        func bounds(_ indices: [Int], in items: [AutoCountDetection]) -> AutoCountBox {
            indices.dropFirst().reduce(items[indices[0]].box) { $0.union(items[$1].box) }
        }
        func locallySupported(_ index: Int) -> Bool {
            let neighbors = detections.indices.filter { $0 != index && distance(detections[index], detections[$0]) <= 2.5 * side }
            guard neighbors.count >= 3 else { return false }
            let xs = (neighbors + [index]).map { detections[$0].box.centerX * aspect }
            let ys = (neighbors + [index]).map { detections[$0].box.centerY }
            return (xs.max()! - xs.min()!) >= 1.25 * side && (ys.max()! - ys.min()!) >= 1.25 * side
        }
        let eligibleText = useText ? textAreas.filter { area in
            guard area.box.isValid, area.confidence.isFinite, area.confidence >= 0.8, area.characterCount >= 2 else { return false }
            let width = area.box.width * aspect, height = area.box.height
            return max(width, height) >= 2 * side && max(width, height) / min(width, height) >= 1.4
        } : []
        var exclusions: [AutoCountRegionResult.Exclusion] = []
        var remaining: [AutoCountDetection] = []
        for (index, detection) in detections.enumerated() {
            let isText = !locallySupported(index) && eligibleText.contains { text in
                text.box.contains(x: detection.box.centerX, y: detection.box.centerY) &&
                text.box.intersectionArea(with: detection.box) / detection.box.area >= 0.7
            }
            if isText { exclusions.append(.init(id: detection.id, reason: .text)) }
            else { remaining.append(detection) }
        }
        guard !remaining.isEmpty else { return result([], exclusions, [], .empty) }
        guard remaining.count >= 6 else { return result(remaining.map(\.id), exclusions, [], .sparse) }
        var visited = Set<Int>(), components: [[Int]] = []
        for seed in remaining.indices where !visited.contains(seed) {
            visited.insert(seed)
            var component = [seed], cursor = 0
            while cursor < component.count {
                let current = component[cursor]
                for other in remaining.indices where !visited.contains(other) {
                    if distance(remaining[current], remaining[other]) <= 3 * side {
                        visited.insert(other); component.append(other)
                    }
                }
                cursor += 1
            }
            components.append(component)
        }
        let supported = components.filter { $0.count >= 3 }
        let reliable = supported.filter { indices in
            let box = bounds(indices, in: remaining)
            return indices.count >= 6 && box.width * aspect >= 2 * side && box.height >= 2 * side
        }
        let supportCount = supported.reduce(0) { $0 + $1.count }
        guard !reliable.isEmpty, Double(supportCount) / Double(remaining.count) >= 0.65 else {
            return result(remaining.map(\.id), exclusions, [], .ambiguous)
        }
        let supportedIndices = supported.flatMap { $0 }
        let supportedBoxes = supported.map { bounds($0, in: remaining) }
        var removed = Set<UUID>()
        if useGeometry {
            for component in components where component.count == 1 {
                let detection = remaining[component[0]]
                guard detection.confidence < 0.7 else { continue }
                let nearest = supportedIndices.map { distance(detection, remaining[$0]) }.min() ?? 0
                let nearBox = supportedBoxes.contains {
                    $0.expanded(horizontal: 3 * side / aspect, vertical: 3 * side).contains(x: detection.box.centerX, y: detection.box.centerY)
                }
                if nearest > 6 * side && !nearBox {
                    removed.insert(detection.id); exclusions.append(.init(id: detection.id, reason: .isolated))
                }
            }
        }
        // Region outlines include retained protruding ends and every supported bundle.
        var regionBoxes = supportedBoxes
        for component in components where component.count < 3 {
            let kept = component.filter { !removed.contains(remaining[$0].id) }
            guard !kept.isEmpty else { continue }
            let box = bounds(kept, in: remaining)
            let nearest = supported.enumerated().map { entry in
                (entry.offset, entry.element.flatMap { a in kept.map { b in distance(remaining[a], remaining[b]) } }.min() ?? .infinity)
            }.min { $0.1 < $1.1 }
            if let nearest, nearest.1 <= 6 * side { regionBoxes[nearest.0] = regionBoxes[nearest.0].union(box) }
            else { regionBoxes.append(box) }
        }
        let regions = regionBoxes.map { $0.expanded(horizontal: 0.5 * side / aspect, vertical: 0.5 * side).clippedToImage }.filter(\.isValid)
        return result(remaining.filter { !removed.contains($0.id) }.map(\.id), exclusions, regions, .none)
    }

    private static func result(_ ids: [UUID], _ excluded: [AutoCountRegionResult.Exclusion], _ regions: [AutoCountBox],
                               _ reason: AutoCountRegionResult.FallbackReason) -> AutoCountRegionResult {
        AutoCountRegionResult(keptIDs: ids, excluded: excluded, regions: regions, fallbackReason: reason)
    }
}
