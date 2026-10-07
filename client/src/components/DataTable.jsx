import { useState, useMemo } from "react";
import { ChevronUp, ChevronDown } from "lucide-react";

export default function DataTable({ columns, data = [], pageSize = 15 }) {
  const [page, setPage] = useState(1);
  const [sortCol, setSortCol] = useState(null);
  const [sortDir, setSortDir] = useState("asc");

  const sorted = useMemo(() => {
    if (!sortCol) return data;
    return [...data].sort((a, b) => {
      const va = a[sortCol] ?? "";
      const vb = b[sortCol] ?? "";
      if (typeof va === "number") return sortDir === "asc" ? va - vb : vb - va;
      return sortDir === "asc" ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
    });
  }, [data, sortCol, sortDir]);

  const totalPages = Math.ceil(sorted.length / pageSize);
  const sliced = sorted.slice((page - 1) * pageSize, page * pageSize);

  const handleSort = (col) => {
    if (sortCol === col) setSortDir((direction) => direction === "asc" ? "desc" : "asc");
    else {
      setSortCol(col);
      setSortDir("asc");
    }
    setPage(1);
  };

  return (
    <div className="data-table-shell">
      <div className="data-table-wrapper">
        <table className="data-table">
          <thead>
            <tr>
              {columns.map((col) => (
                <th key={col.key} aria-sort={sortCol === col.key ? `${sortDir}ending` : "none"}>
                  <button className="table-sort" type="button" onClick={() => handleSort(col.key)}>
                    {col.label}
                    {sortCol === col.key
                      ? (sortDir === "asc" ? <ChevronUp size={12} /> : <ChevronDown size={12} />)
                      : <ChevronUp size={12} className="table-sort-idle" />}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sliced.map((row, index) => (
              <tr key={index}>
                {columns.map((col) => <td key={col.key}>{col.render ? col.render(row[col.key], row) : (row[col.key] ?? "—")}</td>)}
              </tr>
            ))}
            {!sliced.length && <tr><td className="table-empty" colSpan={columns.length}>No data is available for this view.</td></tr>}
          </tbody>
        </table>
      </div>
      {totalPages > 1 && (
        <div className="table-pagination">
          <span>Showing {(page - 1) * pageSize + 1}–{Math.min(page * pageSize, sorted.length)} of {sorted.length}</span>
          <div>
            <button className="btn btn-ghost btn-small" onClick={() => setPage((current) => Math.max(1, current - 1))} disabled={page === 1}>Previous</button>
            <button className="btn btn-ghost btn-small" onClick={() => setPage((current) => Math.min(totalPages, current + 1))} disabled={page === totalPages}>Next</button>
          </div>
        </div>
      )}
    </div>
  );
}
