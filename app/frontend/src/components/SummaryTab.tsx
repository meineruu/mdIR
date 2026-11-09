import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { fetchSummary, type Delta, type Summary } from "../api";
import { fmt, fmtP, fmtSigned } from "../format";

const COLS = ["P@5", "P@10", "R@10", "F1@10", "nDCG@10", "MAP"];
const FINAL = "Final stacking";
const DELTAS = ["P@5", "P@10", "nDCG@10", "MAP"];

export default function SummaryTab() {
  const [data, setData] = useState<Summary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    fetchSummary()
      .then((d) => active && setData(d))
      .catch((e: Error) => active && setError(e.message));
    return () => {
      active = false;
    };
  }, []);

  const rows = useMemo(
    () =>
      data
        ? Object.entries(data.final_table)
            .map(([name, v]) => ({ name, ...v.test }))
            .sort((a, b) => (b as unknown as Record<string, number>)["P@5"] - (a as unknown as Record<string, number>)["P@5"])
        : [],
    [data],
  ) as ({ name: string } & Record<string, number>)[];

  if (error) return <p role="alert" className="rounded-lg bg-red-50 px-4 py-3 text-sm text-red-800">{error}</p>;
  if (!data) return <div className="h-96 animate-pulse rounded-xl bg-line/60" aria-label="Memuat ringkasan" />;

  const fve = data.final_vs_expansion;
  return (
    <div className="space-y-5">
      <section className="rounded-xl bg-surface p-4 shadow-sm sm:p-5">
        <h2 className="font-semibold">Precision@5 pada split test</h2>
        <p className="mb-3 text-sm text-ink-2">Model yang semakin atas semakin tepat pada lima hasil pertama (323 query).</p>
        <div className="h-[420px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 48, bottom: 4, left: 4 }}>
              <CartesianGrid horizontal={false} stroke="#e1e0d9" />
              <XAxis type="number" domain={[0, 0.5]} ticks={[0, 0.1, 0.2, 0.3, 0.4, 0.5]} tick={{ fill: "#6f6d68", fontSize: 12 }} axisLine={{ stroke: "#c3c2b7" }} tickLine={false} tickFormatter={(v: number) => fmt(v, 1)} />
              <YAxis type="category" dataKey="name" width={185} tick={{ fill: "#52514e", fontSize: 12 }} axisLine={{ stroke: "#c3c2b7" }} tickLine={false} />
              <Tooltip cursor={{ fill: "#f4f5f3" }} formatter={(v: number) => [fmt(v, 4), "P@5"]} />
              <Bar dataKey="P@5" isAnimationActive={false} barSize={20} radius={[0, 4, 4, 0]}>
                {rows.map((r) => (
                  <Cell key={r.name} fill={r.name === FINAL ? "#2a78d6" : "#c3c2b7"} />
                ))}
                <LabelList dataKey="P@5" position="right" formatter={(v: number) => fmt(v)} style={{ fill: "#52514e", fontSize: 12 }} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="overflow-x-auto rounded-xl bg-surface p-4 shadow-sm sm:p-5">
        <h2 className="mb-2 font-semibold">Hasil lengkap split test</h2>
        <table className="tnum w-full min-w-[680px] text-sm">
          <thead>
            <tr className="border-b border-line text-left text-ink-2">
              <th className="py-2 font-medium">Metode</th>
              {COLS.map((c) => (
                <th key={c} className="py-2 text-right font-medium">{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.name} className={`border-b border-line/60 last:border-0 ${r.name === FINAL ? "bg-blue-50 font-semibold" : ""}`}>
                <td className="py-2 pl-1">{r.name}</td>
                {COLS.map((c) => (
                  <td key={c} className="py-2 pr-1 text-right">{fmt(r[c], 4)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="overflow-x-auto rounded-xl bg-surface p-4 shadow-sm sm:p-5">
        <h2 className="font-semibold">Uji signifikansi: Final stacking dibanding BM25</h2>
        <p className="mb-2 text-sm text-ink-2">Uji Wilcoxon berpasangan per query, dengan interval kepercayaan 95% dari bootstrap.</p>
        <table className="tnum w-full min-w-[620px] text-sm">
          <thead>
            <tr className="border-b border-line text-left text-ink-2">
              <th className="py-2 font-medium">Metrik</th>
              <th className="py-2 text-right font-medium">Selisih rata-rata</th>
              <th className="py-2 text-right font-medium">CI 95%</th>
              <th className="py-2 text-right font-medium">p Wilcoxon</th>
              <th className="py-2 text-right font-medium">Naik / turun / sama</th>
            </tr>
          </thead>
          <tbody>
            {DELTAS.map((k) => {
              const d = data.error_analysis[`delta_${k}`] as Delta;
              return (
                <tr key={k} className="border-b border-line/60 last:border-0">
                  <td className="py-2">{k}</td>
                  <td className="py-2 text-right">{fmtSigned(d.mean)}</td>
                  <td className="py-2 text-right">{fmt(d.ci95[0])} sampai {fmt(d.ci95[1])}</td>
                  <td className="py-2 text-right">{fmtP(d.wilcoxon_p)}</td>
                  <td className="py-2 text-right">{d.better} / {d.worse} / {d.same}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="mt-3 rounded-lg bg-page px-3 py-2 text-sm text-ink-2">
          Dibanding Expansion+kNN+Co-cite yang lebih sederhana, selisih P@5 Final hanya {fmtSigned(fve.selisih_p5)} (p = {fmtP(fve.p_wilcoxon)}), jadi{" "}
          <strong className="font-semibold text-ink">belum signifikan</strong>. Kenaikan terbesar datang dari sinyal berbasis label train.
        </p>
      </section>
    </div>
  );
}
