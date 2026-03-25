//
//  Changelog.swift
//  Pocketcat
//
//  Created by Amélie Heinrich on 06/03/2026.
//
//  ── HOW TO EDIT ──────────────────────────────────────────────────────────────
//  Add a new ChangelogEntry to `entries` below. Each entry has a version label,
//  a date string, and an array of items. Pick the tag that best describes each
//  item:
//    .added    – new feature or capability
//    .changed  – behaviour or API change
//    .fixed    – bug that has been resolved
//    .known    – known issue / bug still present (useful for Apple reviewers)
//    .note     – neutral remark or context
//  ─────────────────────────────────────────────────────────────────────────────

import SwiftUI

// MARK: - Data model

enum ChangelogTag {
    case added, changed, fixed, known, note
}

struct ChangelogItem {
    let tag: ChangelogTag
    let text: String
}

struct ChangelogEntry {
    let version: String
    let date: String
    let items: [ChangelogItem]
}

// MARK: - *** Edit entries here ***

let changelogEntries: [ChangelogEntry] = [
    ChangelogEntry(
        version: "0.0.1",
        date: "March 2026",
        items: [
            ChangelogItem(tag: .added,   text: "Experimental: RTAO, Stochastic RT shadows, RTGI"),
            ChangelogItem(tag: .known,   text: "Metal4FX temporal upscaling currently has barrier issues that causes flickering and stomps when used"),
            ChangelogItem(tag: .known,   text: "The renderer leaks memory even with an autoreleasepool"),
            ChangelogItem(tag: .known,   text: "Indirect TLAS builds are broken"),
        ]
    ),
]

// MARK: - Tag appearance

private extension ChangelogTag {
    var label: String {
        switch self {
        case .added:   return "Added"
        case .changed: return "Changed"
        case .fixed:   return "Fixed"
        case .known:   return "Known issue"
        case .note:    return "Note"
        }
    }

    var color: Color {
        switch self {
        case .added:   return .green
        case .changed: return .cyan
        case .fixed:   return .blue
        case .known:   return .orange
        case .note:    return .white
        }
    }

    var icon: String {
        switch self {
        case .added:   return "plus.circle.fill"
        case .changed: return "arrow.triangle.2.circlepath"
        case .fixed:   return "checkmark.circle.fill"
        case .known:   return "exclamationmark.triangle.fill"
        case .note:    return "info.circle.fill"
        }
    }
}

// MARK: - Changelog panel view

struct ChangelogPanel: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            VStack(alignment: .leading, spacing: 2) {
                Text("Changelog")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.white)
                Text("Recent engine changes & known issues")
                    .font(.system(size: 10, weight: .regular))
                    .foregroundStyle(.white.opacity(0.35))
            }
            .padding(.horizontal, 16)
            .padding(.top, 20)
            .padding(.bottom, 14)

            Divider()
                .background(.white.opacity(0.08))

            ScrollView(.vertical, showsIndicators: false) {
                VStack(alignment: .leading, spacing: 20) {
                    ForEach(changelogEntries.indices, id: \.self) { i in
                        EntryView(entry: changelogEntries[i])
                    }
                }
                .padding(16)
            }
        }
        .frame(width: 270)
        .background(.white.opacity(0.03))
        .overlay(alignment: .leading) {
            Rectangle()
                .fill(.white.opacity(0.08))
                .frame(width: 1)
        }
    }
}

// MARK: - Single entry

private struct EntryView: View {
    let entry: ChangelogEntry

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text(entry.version)
                    .font(.system(size: 12, weight: .semibold, design: .monospaced))
                    .foregroundStyle(.white)
                Text(entry.date)
                    .font(.system(size: 10, weight: .regular))
                    .foregroundStyle(.white.opacity(0.35))
            }

            VStack(alignment: .leading, spacing: 6) {
                ForEach(entry.items.indices, id: \.self) { i in
                    ItemRow(item: entry.items[i])
                }
            }
        }
    }
}

// MARK: - Single item row

private struct ItemRow: View {
    let item: ChangelogItem

    var body: some View {
        HStack(alignment: .top, spacing: 6) {
            Image(systemName: item.tag.icon)
                .font(.system(size: 9, weight: .semibold))
                .foregroundStyle(item.tag.color)
                .frame(width: 12, height: 14, alignment: .center)
                .padding(.top, 1)

            Text(item.text)
                .font(.system(size: 11, weight: .regular))
                .foregroundStyle(.white.opacity(0.80))
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
