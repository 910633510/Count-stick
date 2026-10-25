import Foundation
import CoreGraphics
import UIKit

/// Positions use the displayed image's top-left origin, independent of screen size.
struct CountMarker: Identifiable, Codable, Equatable, Sendable {
    var id = UUID()
    var x: Double
    var y: Double
    var isManual = false

    init(id: UUID = UUID(), x: Double, y: Double, isManual: Bool = false) {
        self.id = id
        self.x = min(1, max(0, x))
        self.y = min(1, max(0, y))
        self.isManual = isManual
    }
}

struct MarkerEdits {
    private(set) var markers: [CountMarker] = []
    private var undoStack: [[CountMarker]] = []
    var canUndo: Bool { !undoStack.isEmpty }

    mutating func replace(with markers: [CountMarker]) {
        self.markers = markers
        undoStack.removeAll()
    }

    mutating func add(at point: CGPoint) {
        guard point.x.isFinite, point.y.isFinite,
              (0...1).contains(point.x), (0...1).contains(point.y) else { return }
        remember()
        markers.append(CountMarker(x: point.x, y: point.y, isManual: true))
    }

    mutating func remove(id: UUID) {
        guard markers.contains(where: { $0.id == id }) else { return }
        remember()
        markers.removeAll { $0.id == id }
    }

    mutating func undo() {
        guard let previous = undoStack.popLast() else { return }
        markers = previous
    }

    private mutating func remember() {
        undoStack.append(markers)
        if undoStack.count > 50 { undoStack.removeFirst() }
    }
}

enum ImageGeometry {
    static func safeDimension(_ value: CGFloat) -> CGFloat { value.isFinite ? max(0, value) : 0 }

    static func fittedRect(image: CGSize, canvas: CGSize) -> CGRect {
        guard image.width > 0, image.height > 0, canvas.width > 0, canvas.height > 0,
              image.width.isFinite, image.height.isFinite, canvas.width.isFinite, canvas.height.isFinite else { return .zero }
        let scale = min(canvas.width / image.width, canvas.height / image.height)
        let size = CGSize(width: image.width * scale, height: image.height * scale)
        return CGRect(x: (canvas.width - size.width) / 2, y: (canvas.height - size.height) / 2, width: size.width, height: size.height)
    }

    static func normalizedPoint(_ point: CGPoint, in rect: CGRect) -> CGPoint? {
        guard rect.width > 0, rect.height > 0, point.x.isFinite, point.y.isFinite,
              point.x >= rect.minX, point.x <= rect.maxX,
              point.y >= rect.minY, point.y <= rect.maxY else { return nil }
        return CGPoint(x: (point.x - rect.minX) / rect.width, y: (point.y - rect.minY) / rect.height)
    }

    static func nearestMarker(to point: CGPoint, markers: [CountMarker], in rect: CGRect, maximumDistance: CGFloat = 22) -> CountMarker? {
        guard normalizedPoint(point, in: rect) != nil else { return nil }
        let nearest = markers.min { lhs, rhs in
            let a = position(of: lhs, in: rect)
            let b = position(of: rhs, in: rect)
            return hypot(a.x - point.x, a.y - point.y) < hypot(b.x - point.x, b.y - point.y)
        }
        guard let nearest else { return nil }
        let markerPoint = position(of: nearest, in: rect)
        return hypot(markerPoint.x - point.x, markerPoint.y - point.y) <= maximumDistance ? nearest : nil
    }

    static func position(of marker: CountMarker, in rect: CGRect) -> CGPoint {
        CGPoint(x: rect.minX + marker.x * rect.width, y: rect.minY + marker.y * rect.height)
    }
}

enum L10n {
    static func text(_ key: String) -> String { NSLocalizedString(key, comment: "") }
    static func count(_ count: Int) -> String { String(format: text("count.format"), count) }
}

enum AppPalette {
    static let ink = UIColor(red: 0.14, green: 0.18, blue: 0.17, alpha: 1)
    static let coral = UIColor(red: 0.87, green: 0.28, blue: 0.17, alpha: 1)
}
