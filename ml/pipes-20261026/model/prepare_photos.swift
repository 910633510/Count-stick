// Derived-input preparation only. Matches PhotoProcessing.load's ImageIO stage;
// UIKit letterbox rendering is separately supplied/audited by the App test harness.
import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

struct Item: Decodable { let id: String; let source: String; let destination: String }
guard CommandLine.arguments.count == 3 else { fatalError("input-json output-json required") }
let items = try JSONDecoder().decode([Item].self, from: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1])))
var results: [[String: Any]] = []
for item in items {
    let url = URL(fileURLWithPath: item.source)
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: 2048,
            kCGImageSourceShouldCacheImmediately: true
          ] as CFDictionary) else { fatalError("cannot decode \(item.id)") }
    let target = URL(fileURLWithPath: item.destination)
    try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
    guard let output = CGImageDestinationCreateWithURL(target as CFURL, UTType.png.identifier as CFString, 1, nil) else { fatalError("cannot create PNG") }
    CGImageDestinationAddImage(output, image, nil)
    guard CGImageDestinationFinalize(output) else { fatalError("cannot write PNG") }
    results.append(["id": item.id, "source": item.source, "destination": item.destination,
                    "width": image.width, "height": image.height,
                    "operation": "ImageIO EXIF transform + max2048 thumbnail, same options as PhotoProcessing.load"])
}
try JSONSerialization.data(withJSONObject: results, options: [.prettyPrinted, .sortedKeys])
    .write(to: URL(fileURLWithPath: CommandLine.arguments[2]))
