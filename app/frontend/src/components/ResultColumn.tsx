import { useState, type ReactNode } from "react";
import type { Hasil } from "../api";
import { fmt } from "../format";

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

function Card({ item }: { item: Hasil }) {
  const [open, setOpen] = useState(false);
  return (
    <li className="rounded-xl bg-surface p-5 shadow-sm">
      <h3 className="text-lg font-medium leading-snug">
        <span className="mr-1.5 text-ink-2">{item.peringkat}.</span>
        <Highlight text={item.judul} words={item.kata_tampil} />
      </h3>
      {item.relevansi !== null && (
        <div className="mt-2">
          <RelevanceBadge value={item.relevansi} />
        </div>
      )}
      <p className="mt-2 leading-relaxed text-ink-2">
        <Highlight text={item.cuplikan} words={item.kata_tampil} />
      </p>
      {!item.memuat_kata_query && (
        <p className="mt-2 text-sm text-amber-900">Ditemukan walau tidak memuat kata yang kamu cari.</p>
      )}

      {item.sinyal && (
        <div className="mt-3">
          <button onClick={() => setOpen(!open)} aria-expanded={open} className="text-sm font-medium text-brand hover:underline">
            Mengapa dokumen ini muncul {open ? "▲" : "▼"}
          </button>
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
              <p className="tnum mt-1.5 text-xs text-ink-2">
                {item.doc_id} · skor akhir {fmt(item.skor)} · {item.punya_label_train ? "punya label train" : "tidak punya label train"}
              </p>
            </div>
          )}
        </div>
      )}
    </li>
  );
}

export default function ResultColumn({ results }: { results: Hasil[] }) {
  return (
    <section>
      <h2 className="mb-3 text-xl font-semibold">Final stacking</h2>
      <ol className="space-y-3">
        {results.map((r) => (
          <Card key={r.doc_id} item={r} />
        ))}
      </ol>
    </section>
  );
}
