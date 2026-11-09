import { useState, type ReactNode } from "react";
import { Bar, BarChart, Tooltip, XAxis, YAxis, ResponsiveContainer, type TooltipProps } from "recharts";
import type { Hasil, Sinyal } from "../api";
import { fmt, SIGNAL_COLORS } from "../format";

function Highlight({ text, words }: { text: string; words: string[] }): ReactNode {
  if (!words.length) return text;
  const pattern = new RegExp(`\\b(${words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})\\b`, "gi");
  return text.split(pattern).map((part, i) =>
    i % 2 === 1 ? (
      <mark key={i} className="rounded-sm bg-amber-200/70 px-0.5 text-inherit">
        {part}
      </mark>
    ) : (
      part
    ),
  );
}

function RelevanceBadge({ value }: { value: number | null }) {
  if (value === null) return null;
  if (value > 0) {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-green-100 px-2 py-0.5 text-xs font-medium text-green-900">
        <span aria-hidden>✓</span> Relevan · skor {value}
      </span>
    );
  }
  return (
    <span className="inline-flex flex-wrap items-center gap-x-1.5 text-xs text-ink-2">
      <span className="rounded-full bg-line px-2 py-0.5 font-medium">Tidak ada di qrels</span>
      <span>qrels tidak lengkap, belum tentu salah</span>
    </span>
  );
}

function ContributionTooltip({ active, payload }: TooltipProps<number, string>) {
  if (!active || !payload?.length) return null;
  return (
    <div className="tnum rounded-lg bg-white p-2 text-xs shadow-lg ring-1 ring-black/10">
      {payload.map((p) => (
        <div key={p.dataKey as string} className="flex justify-between gap-4 py-0.5">
          <span className="flex items-center gap-1.5 text-ink-2">
            <span className="size-2 rounded-sm" style={{ background: p.color }} />
            {p.name}
          </span>
          <span>{fmt(p.value ?? 0)}</span>
        </div>
      ))}
    </div>
  );
}

function ContributionBar({ signals, max }: { signals: Sinyal[]; max: number }) {
  const row = Object.fromEntries(signals.map((s) => [s.kode, s.kontribusi]));
  return (
    <div className="h-7" role="img" aria-label={signals.map((s) => `${s.nama} ${fmt(s.kontribusi)}`).join(", ")}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={[row]} layout="vertical" margin={{ top: 2, right: 0, bottom: 2, left: 0 }}>
          <XAxis type="number" domain={[0, max]} hide />
          <YAxis type="category" hide />
          <Tooltip content={<ContributionTooltip />} cursor={false} allowEscapeViewBox={{ x: true, y: true }} wrapperStyle={{ zIndex: 20 }} />
          {signals.map((s) => (
            <Bar key={s.kode} dataKey={s.kode} name={s.nama} stackId="a" fill={SIGNAL_COLORS[s.kode]} stroke="#fcfcfb" strokeWidth={2} isAnimationActive={false} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function Card({ item, maxScore }: { item: Hasil; maxScore: number }) {
  const [open, setOpen] = useState(false);
  return (
    <li className="rounded-xl bg-surface p-4 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="font-medium leading-snug">
            <span className="mr-1.5 text-ink-2">{item.peringkat}.</span>
            <Highlight text={item.judul} words={item.kata_tampil} />
          </h3>
          <p className="tnum mt-0.5 text-xs text-ink-2">
            {item.doc_id} · skor {fmt(item.skor)}
          </p>
        </div>
      </div>
      <div className="mt-2">
        <RelevanceBadge value={item.relevansi} />
      </div>
      <p className="mt-2 text-sm leading-relaxed text-ink-2">
        <Highlight text={item.cuplikan} words={item.kata_tampil} />
      </p>
      <p className="mt-2 text-xs text-ink-2">
        {item.memuat_kata_query ? (
          <>
            Memuat kata query: <span className="font-medium text-ink">{item.kata_cocok.join(", ")}</span>
          </>
        ) : (
          <span className="font-medium text-amber-900">Tidak memuat kata query sama sekali (kesenjangan kata)</span>
        )}
      </p>

      {item.sinyal && (
        <div className="mt-3">
          <button onClick={() => setOpen(!open)} aria-expanded={open} className="mb-1 text-xs font-medium text-brand hover:underline">
            Mengapa dokumen ini muncul {open ? "▲" : "▼"}
          </button>
          <ContributionBar signals={item.sinyal} max={maxScore} />
          {open && (
            <div className="mt-2 overflow-x-auto">
              <table className="tnum w-full text-xs">
                <thead>
                  <tr className="border-b border-line text-left text-ink-2">
                    <th className="py-1 font-medium">Sinyal</th>
                    <th className="py-1 text-right font-medium">Bobot</th>
                    <th className="py-1 text-right font-medium">Skor ternormalisasi</th>
                    <th className="py-1 text-right font-medium">Kontribusi</th>
                  </tr>
                </thead>
                <tbody>
                  {item.sinyal.map((s) => (
                    <tr key={s.kode} className="border-b border-line/50 last:border-0">
                      <td className="py-1">{s.nama}</td>
                      <td className="py-1 text-right">{fmt(s.bobot, 2)}</td>
                      <td className="py-1 text-right">{fmt(s.skor_ternormalisasi)}</td>
                      <td className="py-1 text-right font-medium">{fmt(s.kontribusi)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="mt-1.5 text-xs text-ink-2">
                {item.punya_label_train ? "Dokumen ini punya label train." : "Dokumen ini tidak punya label train, jadi sinyal berbasis label lemah."}
              </p>
            </div>
          )}
        </div>
      )}
    </li>
  );
}

export default function ResultColumn({ title, results, final }: { title: string; results: Hasil[]; final: boolean }) {
  const maxScore = Math.max(...results.map((r) => r.skor), 0.01);
  const signals = results[0]?.sinyal ?? [];
  return (
    <section>
      <h2 className="mb-2 font-semibold">{title}</h2>
      {final ? (
        <div className="relative z-10 mb-3 rounded-xl bg-surface p-3 shadow-sm">
          <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2">
            {signals.map((s) => (
              <li key={s.kode} className="flex items-center gap-1.5">
                <span className="size-2.5 rounded-sm" style={{ background: SIGNAL_COLORS[s.kode] }} />
                {s.nama}
              </li>
            ))}
          </ul>
          <p className="mt-2 text-xs text-ink-2">
            Panjang bar = bobot × skor ternormalisasi. Hanya 30 kandidat teratas yang diurutkan ulang. BGE dan MedCPT berbobot 0 secara langsung; pengaruhnya masuk lewat Hybrid RRF.
          </p>
        </div>
      ) : (
        <p className="mb-3 rounded-xl bg-surface p-3 text-xs text-ink-2 shadow-sm">
          BM25 hanya memberi skor pada dokumen yang memuat kata query. Kata yang cocok ditandai kuning.
        </p>
      )}
      <ol className="space-y-3">
        {results.map((r) => (
          <Card key={r.doc_id} item={r} maxScore={maxScore} />
        ))}
      </ol>
    </section>
  );
}
