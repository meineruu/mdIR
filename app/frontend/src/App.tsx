import { useState } from "react";
import SearchTab from "./components/SearchTab";
import SummaryTab from "./components/SummaryTab";

type Tab = "cari" | "ringkasan";

const TABS: { id: Tab; label: string }[] = [
  { id: "cari", label: "Pencarian" },
  { id: "ringkasan", label: "Ringkasan eksperimen" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("cari");
  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-30 bg-surface shadow-sm">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-4 py-3 sm:px-6">
          <div>
            <h1 className="text-xl font-semibold text-brand">MedIR</h1>
            <p className="text-sm text-ink-2">Temu kembali literatur biomedis pada NFCorpus</p>
          </div>
          <nav className="flex gap-1 rounded-lg bg-page p-1" aria-label="Tab utama">
            {TABS.map((t) => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                aria-current={tab === t.id ? "page" : undefined}
                className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                  tab === t.id ? "bg-brand text-white" : "text-ink-2 hover:bg-line"
                }`}
              >
                {t.label}
              </button>
            ))}
          </nav>
        </div>
        <div className="bg-amber-50 text-amber-950" role="note">
          <p className="mx-auto max-w-7xl px-4 py-2 text-xs leading-relaxed sm:px-6 sm:text-sm">
            <strong className="font-semibold">Keterbatasan:</strong> sinyal berbasis label hanya bekerja untuk dokumen
            yang punya label train (99,9% dokumen relevan test memenuhi ini; dokumen baru akan mendekati hasil tanpa
            label). Angka resmi hanya dari split test.
          </p>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6">
        {/* SearchTab tetap terpasang agar hasil tidak hilang saat pindah tab */}
        <div hidden={tab !== "cari"}>
          <SearchTab />
        </div>
        {tab === "ringkasan" && <SummaryTab />}
      </main>
    </div>
  );
}
