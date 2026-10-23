import SwiftUI
import PhotosUI
import AVFoundation
import UIKit

private enum Theme {
    static let ink = Color(red: 0.12, green: 0.19, blue: 0.18)
    static let paper = Color(red: 0.97, green: 0.96, blue: 0.92)
    static let muted = Color(red: 0.39, green: 0.44, blue: 0.41)
    static let gold = Color(red: 0.94, green: 0.69, blue: 0.28)
    static let line = Color(red: 0.86, green: 0.87, blue: 0.82)
    static let coral = Color(uiColor: AppPalette.coral)
}

struct ContentView: View {
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @StateObject private var workspace = CountWorkspace()
    @StateObject private var history = CountHistory()
    @State private var selectedTab = 0
    @State private var selectedPhoto: PhotosPickerItem?
    @State private var showPhotoPicker = false
    @State private var showCamera = false
    @State private var showPermissionAlert = false
    @State private var notice: String?
    @State private var pendingAction: (() -> Void)?
    @State private var confirmReplace = false
    @State private var shareItem: SharePayload?
    @State private var showEditor = false
    @State private var markerMode: MarkerMode = .inspect
    @State private var importing = false
    @FocusState private var noteFocused: Bool

    var body: some View {
        TabView(selection: $selectedTab) {
            NavigationStack {
                ScrollView {
                    VStack(alignment: .leading, spacing: 24) {
                        brandHeader
                        if workspace.phase == .empty && !importing {
                            emptyWorkspace
                        } else {
                            resultWorkspace
                        }
                    }
                    .padding(.horizontal, 22)
                    .padding(.top, 18)
                    .padding(.bottom, 28)
                    .frame(maxWidth: 760)
                    .frame(maxWidth: .infinity)
                }
                .background(Theme.paper)
                .toolbar(.hidden, for: .navigationBar)
                .toolbar {
                    ToolbarItem(placement: .keyboard) {
                        Button(L10n.text("done")) { noteFocused = false }
                    }
                }
            }
            .tabItem { Label(L10n.text("tab.count"), systemImage: "viewfinder") }
            .tag(0)

            HistoryView(history: history) { record in
                guard !importing && workspace.phase != .processing else {
                    notice = L10n.text("processing.wait")
                    return
                }
                runReplacingIfNeeded {
                    guard let image = history.image(for: record) else {
                        notice = L10n.text("error.history.image")
                        return
                    }
                    workspace.restore(record, image: image)
                    markerMode = .inspect
                    selectedTab = 0
                }
            }
            .tabItem { Label(L10n.text("tab.history"), systemImage: "clock.arrow.circlepath") }
            .tag(1)

            GuideView()
                .tabItem { Label(L10n.text("tab.guide"), systemImage: "lightbulb") }
                .tag(2)
        }
        .tint(Theme.ink)
        .preferredColorScheme(.light)
        .photosPicker(isPresented: $showPhotoPicker, selection: $selectedPhoto, matching: .images, photoLibrary: .shared())
        .onChange(of: selectedPhoto) { _, photo in
            guard let photo else { return }
            importing = true
            Task {
                do {
                    guard let data = try await photo.loadTransferable(type: Data.self) else { throw CountError.unreadableImage }
                    markerMode = .inspect
                    await workspace.load(data: data)
                } catch { notice = L10n.text("error.image") }
                importing = false
                selectedPhoto = nil
            }
        }
        .fullScreenCover(isPresented: $showCamera) {
            CameraPicker { image in
                markerMode = .inspect
                Task { await workspace.load(image: image) }
            }.ignoresSafeArea()
        }
        .fullScreenCover(isPresented: $showEditor) {
            if let image = workspace.image {
                MarkerEditor(image: image, workspace: workspace)
            }
        }
        .sheet(item: $shareItem) { item in
            ShareSheet(items: [item.image])
        }
        .confirmationDialog(L10n.text("replace.title"), isPresented: $confirmReplace, titleVisibility: .visible) {
            Button(L10n.text("replace.continue"), role: .destructive) {
                let action = pendingAction
                pendingAction = nil
                action?()
            }
            Button(L10n.text("cancel"), role: .cancel) { pendingAction = nil }
        } message: {
            Text(L10n.text("replace.message"))
        }
        .alert(L10n.text("camera.permission.title"), isPresented: $showPermissionAlert) {
            Button(L10n.text("settings")) { openSettings() }
            Button(L10n.text("cancel"), role: .cancel) {}
        } message: { Text(L10n.text("camera.permission.message")) }
        .alert(L10n.text("notice.title"), isPresented: Binding(get: { notice != nil }, set: { if !$0 { notice = nil } })) {
            Button(L10n.text("ok"), role: .cancel) { notice = nil }
        } message: { Text(notice ?? "") }
        .onChange(of: history.records) { previous, records in
            if previous.contains(where: { $0.id == workspace.id }) && !records.contains(where: { $0.id == workspace.id }) {
                workspace.savedRecordWasDeleted()
            }
        }
        .onChange(of: workspace.errorMessage) { _, message in
            if workspace.phase == .empty, let message { notice = message }
        }
        .task {
            #if DEBUG
            // UI automation may supply an actual image. It always runs the real model.
            let arguments = ProcessInfo.processInfo.arguments
            if let index = arguments.firstIndex(of: "--countstick-test-photo"), arguments.indices.contains(index + 1),
               let data = try? Data(contentsOf: URL(fileURLWithPath: arguments[index + 1])) {
                await workspace.load(data: data)
            }
            #endif
        }
    }

    private var brandHeader: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 5) {
                Text("COUNT STICK")
                    .font(.system(size: 13, weight: .bold, design: .rounded))
                    .tracking(2.5)
                    .foregroundStyle(Theme.muted)
                Text(L10n.text(workspace.image == nil ? "home.title" : "result.title"))
                    .font(.system(.largeTitle, design: .rounded, weight: .bold))
                    .foregroundStyle(Theme.ink)
            }
            Spacer(minLength: 8)
            Image(systemName: "dot.viewfinder")
                .font(.system(size: 26, weight: .medium))
                .foregroundStyle(Theme.ink)
                .frame(width: 54, height: 54)
                .background(Theme.gold, in: RoundedRectangle(cornerRadius: 18))
                .accessibilityHidden(true)
        }
    }

    private var emptyWorkspace: some View {
        VStack(alignment: .leading, spacing: 22) {
            VStack(alignment: .leading, spacing: 18) {
                HStack {
                    Label(L10n.text("home.offline"), systemImage: "iphone.gen3.radiowaves.left.and.right.slash")
                        .font(.caption.weight(.semibold))
                        .padding(.horizontal, 12).padding(.vertical, 8)
                        .background(.white.opacity(0.1), in: Capsule())
                    Spacer()
                    Image(systemName: "sparkle").font(.headline).foregroundStyle(Theme.gold).accessibilityHidden(true)
                }
                StickIllustration()
                    .frame(height: 164)
                    .accessibilityHidden(true)
                Text(L10n.text("home.headline"))
                    .font(.system(.title, design: .rounded, weight: .bold))
                    .fixedSize(horizontal: false, vertical: true)
                Text(L10n.text("home.description"))
                    .font(.subheadline)
                    .foregroundStyle(.white.opacity(0.76))
                    .fixedSize(horizontal: false, vertical: true)
            }
            .foregroundStyle(.white)
            .padding(24)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Theme.ink, in: RoundedRectangle(cornerRadius: 28))

            photoActions
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: "checkmark.shield")
                    .font(.title3).foregroundStyle(Theme.ink)
                VStack(alignment: .leading, spacing: 4) {
                    Text(L10n.text("home.private.title")).font(.subheadline.weight(.semibold))
                    Text(L10n.text("home.private.detail")).font(.footnote).foregroundStyle(Theme.muted)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
            .background(.white.opacity(0.7), in: RoundedRectangle(cornerRadius: 18))
        }
    }

    private var photoActions: some View {
        VStack(spacing: 12) {
            Button { runReplacingIfNeeded { openCamera() } } label: {
                Label(L10n.text("action.camera"), systemImage: "camera.fill")
                    .frame(maxWidth: .infinity).padding(.vertical, 17)
            }
            .buttonStyle(PrimaryButtonStyle())
            .accessibilityIdentifier("takePhoto")
            Button { runReplacingIfNeeded { showPhotoPicker = true } } label: {
                Label(L10n.text("action.photos"), systemImage: "photo.on.rectangle")
                    .frame(maxWidth: .infinity).padding(.vertical, 16)
            }
            .font(.body.weight(.semibold))
            .background(.white, in: RoundedRectangle(cornerRadius: 17))
            .overlay(RoundedRectangle(cornerRadius: 17).stroke(Theme.line, lineWidth: 1))
            .accessibilityIdentifier("choosePhoto")
        }
        .disabled(workspace.phase == .processing || importing)
    }

    private var resultWorkspace: some View {
        VStack(alignment: .leading, spacing: 18) {
            if workspace.phase == .processing || importing {
                HStack(spacing: 14) {
                    ProgressView().tint(Theme.ink)
                    VStack(alignment: .leading, spacing: 5) {
                        Text(L10n.text("processing.title")).font(.headline)
                        Text(L10n.text("processing.detail")).font(.footnote).foregroundStyle(Theme.muted)
                    }
                }.padding(20).frame(maxWidth: .infinity, alignment: .leading)
                    .background(.white, in: RoundedRectangle(cornerRadius: 22))
                    .accessibilityIdentifier("processingState")
            }
            if workspace.phase == .ready {
                countSummary
            }
            if let image = workspace.image {
                VStack(spacing: 0) {
                    MarkerCanvas(image: image, markers: workspace.edits.markers, mode: markerMode,
                                 onAdd: workspace.add, onRemove: workspace.remove)
                        .frame(height: horizontalSizeClass == .regular
                               ? min(560, max(300, 700 * image.size.height / max(image.size.width, 1)))
                               : min(420, max(240, 340 * image.size.height / max(image.size.width, 1))))
                        .background(Color.black.opacity(0.04))
                        .clipShape(RoundedRectangle(cornerRadius: 22))
                        .accessibilityIdentifier("resultPhoto")
                    if workspace.phase == .ready {
                        HStack {
                            Label(L10n.text("result.markers"), systemImage: "circle.fill")
                                .font(.caption).foregroundStyle(Theme.muted)
                            Spacer()
                            Button { showEditor = true } label: {
                                Label(L10n.text("action.enlarge"), systemImage: "arrow.up.left.and.arrow.down.right")
                                    .font(.caption.weight(.semibold))
                            }
                            .padding(.vertical, 14)
                            .accessibilityIdentifier("enlargePhoto")
                        }
                    }
                }
            }
            if workspace.phase == .failed {
                VStack(alignment: .leading, spacing: 12) {
                    Label(L10n.text("error.detect.title"), systemImage: "exclamationmark.triangle")
                        .font(.headline)
                    Text(workspace.errorMessage ?? L10n.text("error.detect"))
                        .font(.subheadline).foregroundStyle(Theme.muted)
                    Button(L10n.text("action.retry")) { Task { await workspace.retry() } }
                        .font(.headline)
                    Button(L10n.text("action.manual")) { workspace.countManually(); markerMode = .add }
                        .font(.headline)
                }.padding(20).background(.white, in: RoundedRectangle(cornerRadius: 20))
            }
            if workspace.phase == .ready {
                correctionControls
                VStack(alignment: .leading, spacing: 12) {
                    TextField(L10n.text("note.placeholder"), text: $workspace.note, axis: .vertical)
                        .lineLimit(1...3)
                        .focused($noteFocused)
                        .font(.body)
                        .padding(15)
                        .background(.white, in: RoundedRectangle(cornerRadius: 15))
                        .onChange(of: workspace.note) { _, value in if value.count > 120 { workspace.note = String(value.prefix(120)) } }
                        .accessibilityIdentifier("resultNote")
                    ViewThatFits(in: .horizontal) {
                        HStack(spacing: 12) { saveButton; shareButton }
                        VStack(spacing: 12) { saveButton; shareButton }
                    }
                    Text(L10n.text("result.save.detail"))
                        .font(.caption).foregroundStyle(Theme.muted)
                }
            }
            if workspace.phase != .processing && !importing {
                Button {
                    runReplacingIfNeeded { workspace.reset(); markerMode = .inspect }
                } label: {
                    Label(L10n.text("action.new"), systemImage: "plus")
                        .font(.body.weight(.semibold)).frame(maxWidth: .infinity).padding(.vertical, 12)
                }.accessibilityIdentifier("newCount")
            }
        }
    }

    private var countSummary: some View {
        HStack(alignment: .center, spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                Text(L10n.text("result.total")).font(.subheadline.weight(.semibold)).foregroundStyle(Theme.muted)
                Text(workspace.edits.markers.count.formatted())
                    .font(.system(size: 64, weight: .bold, design: .rounded))
                    .monospacedDigit().minimumScaleFactor(0.5).lineLimit(1)
                    .contentTransition(.numericText())
                    .accessibilityIdentifier("resultCount")
            }
            Spacer(minLength: 0)
            VStack(alignment: .trailing, spacing: 8) {
                Image(systemName: "checkmark.circle").font(.system(size: 28)).foregroundStyle(Theme.ink)
                Text(L10n.text(workspace.automaticCount == nil ? "result.manual" : "result.review"))
                    .font(.subheadline.weight(.semibold)).multilineTextAlignment(.trailing)
                if let automatic = workspace.automaticCount, automatic != workspace.edits.markers.count {
                    Text(String(format: L10n.text("result.initial"), automatic))
                        .font(.caption).foregroundStyle(Theme.muted)
                }
            }.frame(maxWidth: 175)
        }
        .foregroundStyle(Theme.ink).padding(22)
        .background(Theme.gold.opacity(0.28), in: RoundedRectangle(cornerRadius: 23))
        .accessibilityElement(children: .contain)
    }

    private var correctionControls: some View {
        VStack(alignment: .leading, spacing: 13) {
            HStack {
                Text(L10n.text("correction.title")).font(.headline)
                Spacer()
                Button { workspace.undo() } label: { Label(L10n.text("action.undo"), systemImage: "arrow.uturn.backward") }
                    .font(.subheadline.weight(.semibold)).disabled(!workspace.edits.canUndo)
                    .accessibilityIdentifier("undoCorrection")
            }
            Picker(L10n.text("correction.mode"), selection: $markerMode) {
                ForEach(MarkerMode.allCases) { mode in Text(L10n.text(mode.key)).tag(mode) }
            }.pickerStyle(.segmented).accessibilityIdentifier("correctionMode")
            Text(L10n.text(workspace.edits.markers.isEmpty && markerMode == .inspect ? "result.zero" : markerMode.instructionKey))
                .font(.footnote).foregroundStyle(Theme.muted).fixedSize(horizontal: false, vertical: true)
        }
    }

    private var saveButton: some View {
        Button {
            do {
                if let image = workspace.image {
                    try history.save(workspace.record, image: image)
                    workspace.markSaved()
                    UINotificationFeedbackGenerator().notificationOccurred(.success)
                }
            } catch { notice = L10n.text("error.history.save") }
        } label: {
            Label(L10n.text(workspace.isSaved ? "action.saved" : "action.save"), systemImage: workspace.isSaved ? "checkmark" : "tray.and.arrow.down")
                .frame(maxWidth: .infinity).padding(.vertical, 16)
        }.buttonStyle(PrimaryButtonStyle()).disabled(workspace.isSaved)
            .accessibilityIdentifier("saveCount")
    }

    private var shareButton: some View {
        Button {
            if let image = workspace.image {
                shareItem = SharePayload(image: PhotoProcessing.shareImage(photo: image, markers: workspace.edits.markers, note: workspace.note))
            }
        } label: {
            Label(L10n.text("action.share"), systemImage: "square.and.arrow.up")
                .font(.body.weight(.semibold)).frame(maxWidth: .infinity).padding(.vertical, 16)
        }.background(.white, in: RoundedRectangle(cornerRadius: 17))
            .accessibilityIdentifier("shareCount")
    }

    private func runReplacingIfNeeded(_ action: @escaping () -> Void) {
        if workspace.hasUnsavedChanges && workspace.phase == .ready {
            pendingAction = action
            confirmReplace = true
        } else { action() }
    }

    private func openCamera() {
        guard UIImagePickerController.isSourceTypeAvailable(.camera) else {
            notice = L10n.text("camera.unavailable")
            return
        }
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: showCamera = true
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                Task { @MainActor in
                    if granted { showCamera = true } else { showPermissionAlert = true }
                }
            }
        case .denied, .restricted: showPermissionAlert = true
        @unknown default: showPermissionAlert = true
        }
    }
    private func openSettings() {
        guard let url = URL(string: UIApplication.openSettingsURLString) else { return }
        UIApplication.shared.open(url)
    }
}

private struct SharePayload: Identifiable {
    let id = UUID()
    let image: UIImage
}

private struct PrimaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label.font(.body.weight(.semibold)).foregroundStyle(.white)
            .background(Theme.ink.opacity(configuration.isPressed ? 0.78 : 1), in: RoundedRectangle(cornerRadius: 17))
    }
}

private struct StickIllustration: View {
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                RoundedRectangle(cornerRadius: 27).stroke(.white.opacity(0.13), lineWidth: 1)
                    .frame(width: 210, height: 146)
                ForEach(0..<7) { index in
                    Capsule().fill(index.isMultiple(of: 2) ? Theme.gold : Color(red: 0.95, green: 0.85, blue: 0.63))
                        .frame(width: 11, height: CGFloat(108 + (index % 3) * 9))
                        .rotationEffect(.degrees(Double(index - 3) * 5))
                        .offset(x: CGFloat(index - 3) * 21, y: CGFloat(index % 2) * 5)
                }
                Image(systemName: "viewfinder").font(.system(size: 152, weight: .ultraLight))
                    .foregroundStyle(.white.opacity(0.35))
            }.frame(width: ImageGeometry.safeDimension(geometry.size.width), height: ImageGeometry.safeDimension(geometry.size.height))
        }
    }
}

enum MarkerMode: String, CaseIterable, Identifiable {
    case inspect, add, remove
    var id: String { rawValue }
    var key: String { "mode." + rawValue }
    var instructionKey: String { "mode." + rawValue + ".detail" }
}

private struct MarkerCanvas: View {
    let image: UIImage
    let markers: [CountMarker]
    let mode: MarkerMode
    let onAdd: (CGPoint) -> Void
    let onRemove: (UUID) -> Void
    var body: some View {
        GeometryReader { geometry in
            let rect = ImageGeometry.fittedRect(image: image.size, canvas: geometry.size)
            ZStack(alignment: .topLeading) {
                Image(uiImage: image).resizable().scaledToFit()
                    .frame(width: ImageGeometry.safeDimension(geometry.size.width), height: ImageGeometry.safeDimension(geometry.size.height))
                    .contentShape(Rectangle())
                    .gesture(SpatialTapGesture().onEnded { value in
                        if mode == .add, let point = ImageGeometry.normalizedPoint(value.location, in: rect) {
                            onAdd(point)
                            UIImpactFeedbackGenerator(style: .light).impactOccurred()
                        } else if mode == .remove,
                                  let marker = ImageGeometry.nearestMarker(to: value.location, markers: markers, in: rect) {
                            onRemove(marker.id)
                            UIImpactFeedbackGenerator(style: .light).impactOccurred()
                        }
                    })
                    .accessibilityLabel(L10n.text("photo.accessibility"))
                ForEach(Array(markers.enumerated()), id: \.element.id) { index, marker in
                    let point = ImageGeometry.position(of: marker, in: rect)
                    Button {
                        if mode == .remove { onRemove(marker.id) }
                    } label: {
                        ZStack {
                            Circle().fill(Theme.coral).frame(width: 13, height: 13)
                            Circle().stroke(.white, lineWidth: 2).frame(width: 13, height: 13)
                            if mode == .remove {
                                Image(systemName: "minus").font(.system(size: 8, weight: .heavy)).foregroundStyle(.white)
                            }
                        }.frame(width: 30, height: 30).contentShape(Circle())
                    }
                    .buttonStyle(.plain)
                    .position(point)
                    .allowsHitTesting(false)
                    .accessibilityLabel(String(format: L10n.text("marker.accessibility"), index + 1))
                    .accessibilityHint(L10n.text("marker.remove.hint"))
                    .accessibilityHidden(mode != .remove)
                }
            }
            .clipped()
        }
    }
}

private struct MarkerEditor: View {
    let image: UIImage
    @ObservedObject var workspace: CountWorkspace
    @Environment(\.dismiss) private var dismiss
    @State private var mode: MarkerMode = .inspect
    @State private var zoom: CGFloat = 1
    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                Text(L10n.count(workspace.edits.markers.count))
                    .font(.system(.title2, design: .rounded, weight: .bold))
                GeometryReader { geometry in
                    ScrollView([.horizontal, .vertical]) {
                        MarkerCanvas(image: image, markers: workspace.edits.markers, mode: mode, onAdd: workspace.add, onRemove: workspace.remove)
                            .frame(width: ImageGeometry.safeDimension(geometry.size.width * zoom), height: ImageGeometry.safeDimension(geometry.size.height * zoom))
                    }
                    .background(Color.black.opacity(0.04))
                    .clipShape(RoundedRectangle(cornerRadius: 20))
                }
                Picker(L10n.text("correction.mode"), selection: $mode) {
                    ForEach(MarkerMode.allCases) { item in Text(L10n.text(item.key)).tag(item) }
                }.pickerStyle(.segmented)
                Text(L10n.text(mode.instructionKey)).font(.footnote).foregroundStyle(Theme.muted)
                    .frame(maxWidth: .infinity, alignment: .leading)
                HStack {
                    Button { zoom = zoom == 1 ? 2 : 1 } label: {
                        Label(zoom == 1 ? "2×" : "1×", systemImage: zoom == 1 ? "plus.magnifyingglass" : "minus.magnifyingglass")
                    }.accessibilityLabel(L10n.text("action.zoom"))
                    Spacer()
                    Button { workspace.undo() } label: { Label(L10n.text("action.undo"), systemImage: "arrow.uturn.backward") }
                        .disabled(!workspace.edits.canUndo)
                }.font(.body.weight(.semibold)).padding(.vertical, 8)
            }
            .padding(20).background(Theme.paper)
            .navigationTitle(L10n.text("correction.title"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button(L10n.text("done")) { dismiss() } } }
        }.tint(Theme.ink)
    }
}

private struct HistoryView: View {
    @ObservedObject var history: CountHistory
    let onOpen: (CountRecord) -> Void
    @State private var pendingDelete: CountRecord?
    @State private var notice: String?
    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    Text(L10n.text("history.subtitle"))
                        .font(.subheadline).foregroundStyle(Theme.muted)
                    if let loadError = history.errorMessage {
                        Label(loadError, systemImage: "exclamationmark.triangle").font(.subheadline)
                    }
                    if history.records.isEmpty {
                        VStack(spacing: 16) {
                            Image(systemName: "tray").font(.system(size: 48, weight: .light)).foregroundStyle(Theme.muted)
                            Text(L10n.text("history.empty.title")).font(.title3.weight(.semibold))
                            Text(L10n.text("history.empty.detail")).font(.subheadline).foregroundStyle(Theme.muted).multilineTextAlignment(.center)
                        }.frame(maxWidth: .infinity).padding(.vertical, 70)
                    }
                    ForEach(history.records) { record in
                        HStack(spacing: 14) {
                            Button { onOpen(record) } label: {
                                HStack(spacing: 14) {
                                    Group {
                                        if let image = history.image(for: record, thumbnail: true) {
                                            Image(uiImage: image).resizable().scaledToFill()
                                        } else { Image(systemName: "photo").foregroundStyle(Theme.muted) }
                                    }.frame(width: 76, height: 82).background(Theme.paper).clipShape(RoundedRectangle(cornerRadius: 13))
                                    VStack(alignment: .leading, spacing: 5) {
                                        Text(L10n.count(record.count)).font(.system(.title3, design: .rounded, weight: .bold))
                                        if !record.note.isEmpty { Text(record.note).font(.subheadline).lineLimit(2) }
                                        Text(record.date.formatted(date: .abbreviated, time: .shortened))
                                            .font(.caption).foregroundStyle(Theme.muted)
                                    }
                                    Spacer(minLength: 0)
                                }.contentShape(Rectangle())
                            }.buttonStyle(.plain)
                            Menu {
                                Button(role: .destructive) { pendingDelete = record } label: { Label(L10n.text("delete"), systemImage: "trash") }
                            } label: { Image(systemName: "ellipsis").frame(width: 34, height: 44) }
                                .accessibilityLabel(L10n.text("history.options"))
                        }.padding(13).background(.white, in: RoundedRectangle(cornerRadius: 21))
                    }
                }.padding(22).frame(maxWidth: 760).frame(maxWidth: .infinity)
            }
            .background(Theme.paper)
            .navigationTitle(L10n.text("tab.history"))
        }
        .confirmationDialog(L10n.text("history.delete.title"), isPresented: Binding(get: { pendingDelete != nil }, set: { if !$0 { pendingDelete = nil } }), titleVisibility: .visible) {
            Button(L10n.text("delete"), role: .destructive) {
                if let record = pendingDelete {
                    do { try history.delete(record) } catch { notice = L10n.text("error.history.delete") }
                }
                pendingDelete = nil
            }
            Button(L10n.text("cancel"), role: .cancel) { pendingDelete = nil }
        } message: { Text(L10n.text("history.delete.message")) }
        .alert(L10n.text("notice.title"), isPresented: Binding(get: { notice != nil }, set: { if !$0 { notice = nil } })) {
            Button(L10n.text("ok"), role: .cancel) { notice = nil }
        } message: { Text(notice ?? "") }
    }
}

private struct GuideView: View {
    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    Text(L10n.text("guide.subtitle")).font(.title3).foregroundStyle(Theme.muted)
                    VStack(spacing: 22) {
                        guideRow(number: "01", title: "guide.photo.title", body: "guide.photo.detail", symbol: "camera")
                        guideRow(number: "02", title: "guide.review.title", body: "guide.review.detail", symbol: "hand.tap")
                        guideRow(number: "03", title: "guide.save.title", body: "guide.save.detail", symbol: "square.and.arrow.up")
                    }.padding(22).background(.white, in: RoundedRectangle(cornerRadius: 25))
                    VStack(alignment: .leading, spacing: 12) {
                        Label(L10n.text("guide.privacy.title"), systemImage: "lock.shield").font(.headline)
                        Text(L10n.text("guide.privacy.detail")).font(.subheadline).foregroundStyle(Theme.muted)
                    }.padding(22).frame(maxWidth: .infinity, alignment: .leading)
                        .background(Theme.gold.opacity(0.22), in: RoundedRectangle(cornerRadius: 23))
                    VStack(spacing: 0) {
                        Link(destination: URL(string: "https://www.xmaillc.com/count-stick-support.html")!) {
                            linkRow("guide.support", symbol: "questionmark.circle")
                        }
                        Divider()
                        Link(destination: URL(string: "https://www.xmaillc.com/count-stick-privacy.html")!) {
                            linkRow("guide.policy", symbol: "hand.raised")
                        }
                    }.padding(.horizontal, 17).background(.white, in: RoundedRectangle(cornerRadius: 20))
                    Text("Count Stick · \(Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "1.0")")
                        .font(.caption).foregroundStyle(Theme.muted).frame(maxWidth: .infinity)
                }.padding(22).frame(maxWidth: 760).frame(maxWidth: .infinity)
            }.background(Theme.paper).navigationTitle(L10n.text("tab.guide"))
        }
    }
    private func guideRow(number: String, title: String, body: String, symbol: String) -> some View {
        HStack(alignment: .top, spacing: 16) {
            Text(number).font(.system(.headline, design: .monospaced)).foregroundStyle(Theme.muted)
                .frame(width: 32).padding(.top, 2)
            VStack(alignment: .leading, spacing: 7) {
                Label(L10n.text(title), systemImage: symbol).font(.headline)
                Text(L10n.text(body)).font(.subheadline).foregroundStyle(Theme.muted).fixedSize(horizontal: false, vertical: true)
            }
        }.frame(maxWidth: .infinity, alignment: .leading)
    }
    private func linkRow(_ key: String, symbol: String) -> some View {
        HStack {
            Label(L10n.text(key), systemImage: symbol)
            Spacer()
            Image(systemName: "arrow.up.right").font(.caption)
        }.font(.subheadline.weight(.semibold)).padding(.vertical, 17)
    }
}

#Preview { ContentView() }
