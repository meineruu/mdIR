import { useState } from "react";
import SearchTab from "./components/SearchTab";
import SummaryTab from "./components/SummaryTab";

type Tab = "cari" | "ringkasan";

const TABS: { id: Tab; label: string }[] = [
  { id: "cari", label: "Pencarian" },
  { id: "ringkasan", label: "Ringkasan eksperimen" },
];

function Logo() {
  return (
    <svg viewBox="0 0 24 24" className="size-7 text-brand" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M12 6.5C10 4.8 7 4.3 3.5 4.8v13.7c3.5-.5 6.5 0 8.5 1.7 2-1.7 5-2.2 8.5-1.7V4.8C17 4.3 14 4.8 12 6.5Z" />
      <path d="M12 6.5v13.7" />
    </svg>
  );
}

export default function App() {
  const [tab, setTab] = useState<Tab>("cari");
  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-30 bg-white/95 backdrop-blur">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-4 py-3 sm:px-6">
          <div className="flex items-center gap-2">
            <Logo />
            <div>
              <h1 className="text-xl font-semibold leading-tight">MedIR</h1>
              <p className="text-xs text-ink-2">Temu kembali literatur biomedis</p>
            </div>
          </div>
          <nav className="flex items-center gap-1" aria-label="Tab utama">
            {TABS.map((t) => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                aria-current={tab === t.id ? "page" : undefined}
                className={`rounded-full px-4 py-2 text-sm font-medium transition-colors ${
                  tab === t.id ? "btn-glow bg-brand text-white" : "text-ink-2 hover:text-ink"
                }`}
              >
                {t.label}
              </button>
            ))}
          </nav>
        </div>
        <div className="border-y border-line bg-blue-50/70" role="note">
          <p className="mx-auto max-w-6xl px-4 py-1.5 text-xs leading-relaxed text-ink-2 sm:px-6">
            <strong className="font-semibold text-ink">Keterbatasan:</strong> sinyal berbasis label hanya bekerja untuk
            dokumen yang punya label train (99,9% dokumen relevan test memenuhi ini; dokumen baru akan mendekati hasil
            tanpa label). Angka resmi hanya dari split test.
          </p>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-4 py-6 sm:px-6">
        {/* SearchTab tetap terpasang agar hasil tidak hilang saat pindah tab */}
        <div hidden={tab !== "cari"}>
          <SearchTab />
        </div>
        {tab === "ringkasan" && <SummaryTab />}
      </main>
    </div>
  );
}
