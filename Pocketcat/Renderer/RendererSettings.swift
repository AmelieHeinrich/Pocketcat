//
//  RendererSettings.swift
//  Pocketcat
//
//  Created by Amélie Heinrich on 02/03/2026.
//

import Combine
import Foundation
import SwiftUI

enum RendererTimelineType: Int, CaseIterable {
    case Mobile = 0      // Forward, no raytracing
    case Desktop = 1     // Deferred, full raytracing
    case Pathtraced = 2  // Pathtraced reference
}

// MARK: - Storage types

enum SettingValue {
    case bool(Bool)
    case int(Int)
    case float(Float)
    case enumCase(Int, [String])  // selectedIndex, caseLabels
}

enum SettingMetadata {
    case bool
    case int(range: ClosedRange<Int>)
    case float(range: ClosedRange<Float>, step: Float)
    case enumType(caseCount: Int)
}

struct SettingEntry {
    let key: String      // "Tonemap.Gamma"
    let label: String    // "Gamma"
    let section: String  // "Tonemap"
    var value: SettingValue
    let metadata: SettingMetadata
}

// MARK: - SettingsRegistry

final class SettingsRegistry: ObservableObject {
    private var entries: [String: SettingEntry] = [:]
    private var insertionOrder: [String] = []

    // MARK: - Registration (idempotent — first call wins)

    func register(bool key: String, label: String? = nil, default value: Bool) {
        guard entries[key] == nil else { return }
        let entry = SettingEntry(
            key: key, label: label ?? labelFromKey(key), section: sectionFromKey(key),
            value: .bool(value), metadata: .bool)
        insert(entry)
    }

    func register(int key: String, label: String? = nil, default value: Int, range: ClosedRange<Int>) {
        guard entries[key] == nil else { return }
        let entry = SettingEntry(
            key: key, label: label ?? labelFromKey(key), section: sectionFromKey(key),
            value: .int(value), metadata: .int(range: range))
        insert(entry)
    }

    func register(float key: String, label: String? = nil, default value: Float,
                  range: ClosedRange<Float>, step: Float = 0.01) {
        guard entries[key] == nil else { return }
        let entry = SettingEntry(
            key: key, label: label ?? labelFromKey(key), section: sectionFromKey(key),
            value: .float(value), metadata: .float(range: range, step: step))
        insert(entry)
    }

    func register<E>(enum key: String, label: String? = nil, default value: E)
        where E: CaseIterable & RawRepresentable, E.RawValue == Int
    {
        guard entries[key] == nil else { return }
        let labels = E.allCases.map { String(describing: $0) }
        let entry = SettingEntry(
            key: key, label: label ?? labelFromKey(key), section: sectionFromKey(key),
            value: .enumCase(value.rawValue, labels),
            metadata: .enumType(caseCount: labels.count))
        insert(entry)
    }

    func register(dynamicPicker key: String, label: String? = nil) {
        guard entries[key] == nil else { return }
        let entry = SettingEntry(
            key: key, label: label ?? labelFromKey(key), section: sectionFromKey(key),
            value: .enumCase(0, ["(none)"]),
            metadata: .enumType(caseCount: 1))
        insert(entry)
    }

    // Call from main thread only. Clamps selection if options shrink.
    func updatePickerOptions(_ key: String, options: [String]) {
        guard case .enumCase(let idx, _) = entries[key]?.value else { return }
        let labels = options.isEmpty ? ["(none)"] : options
        let clampedIdx = min(idx, labels.count - 1)
        entries[key]?.value = .enumCase(clampedIdx, labels)
        objectWillChange.send()
    }

    // MARK: - Read

    func bool(_ key: String, default fallback: Bool = false) -> Bool {
        guard case .bool(let v) = entries[key]?.value else { return fallback }
        return v
    }

    func int(_ key: String, default fallback: Int = 0) -> Int {
        guard case .int(let v) = entries[key]?.value else { return fallback }
        return v
    }

    func float(_ key: String, default fallback: Float = 0) -> Float {
        guard case .float(let v) = entries[key]?.value else { return fallback }
        return v
    }

    func `enum`<E>(_ key: String, as type: E.Type, default fallback: E) -> E
        where E: CaseIterable & RawRepresentable, E.RawValue == Int
    {
        guard case .enumCase(let idx, _) = entries[key]?.value,
              let v = E(rawValue: idx) else { return fallback }
        return v
    }

    func pickerIndex(_ key: String, default fallback: Int = 0) -> Int {
        guard case .enumCase(let idx, _) = entries[key]?.value else { return fallback }
        return idx
    }

    // MARK: - Bindings

    func binding(bool key: String) -> Binding<Bool> {
        Binding(
            get: { [weak self] in self?.bool(key) ?? false },
            set: { [weak self] newValue in
                guard let self else { return }
                self.objectWillChange.send()
                self.entries[key]?.value = .bool(newValue)
            }
        )
    }

    func binding(float key: String) -> Binding<Float> {
        Binding(
            get: { [weak self] in self?.float(key) ?? 0 },
            set: { [weak self] newValue in
                guard let self else { return }
                self.objectWillChange.send()
                self.entries[key]?.value = .float(newValue)
            }
        )
    }

    func binding(int key: String) -> Binding<Int> {
        Binding(
            get: { [weak self] in self?.int(key) ?? 0 },
            set: { [weak self] newValue in
                guard let self else { return }
                self.objectWillChange.send()
                self.entries[key]?.value = .int(newValue)
            }
        )
    }

    func bindingIndex(_ key: String) -> Binding<Int> {
        Binding(
            get: { [weak self] in
                guard case .enumCase(let idx, _) = self?.entries[key]?.value else { return 0 }
                return idx
            },
            set: { [weak self] newIdx in
                guard let self else { return }
                guard case .enumCase(_, let labels) = self.entries[key]?.value else { return }
                self.objectWillChange.send()
                self.entries[key]?.value = .enumCase(newIdx, labels)
            }
        )
    }

    // MARK: - Section grouping for UI

    func orderedSections() -> [(String, [String])] {
        var result: [(String, [String])] = []
        var sectionIndex: [String: Int] = [:]

        for key in insertionOrder {
            guard let entry = entries[key] else { continue }
            let section = entry.section
            if let idx = sectionIndex[section] {
                result[idx].1.append(key)
            } else {
                sectionIndex[section] = result.count
                result.append((section, [key]))
            }
        }
        return result
    }

    func entry(for key: String) -> SettingEntry? {
        entries[key]
    }

    // MARK: - Private helpers

    private func insert(_ entry: SettingEntry) {
        entries[entry.key] = entry
        insertionOrder.append(entry.key)
        objectWillChange.send()
    }

    private func sectionFromKey(_ key: String) -> String {
        guard let dot = key.firstIndex(of: ".") else { return key }
        return String(key[key.startIndex..<dot])
    }

    private func labelFromKey(_ key: String) -> String {
        guard let dot = key.firstIndex(of: ".") else { return key }
        return String(key[key.index(after: dot)...])
    }
}
