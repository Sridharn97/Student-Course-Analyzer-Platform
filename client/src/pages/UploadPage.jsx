import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  AlertTriangle,
  ArrowRight,
  BarChart3,
  Brain,
  CheckCircle,
  FileSpreadsheet,
  GraduationCap,
  Sparkles,
} from "lucide-react";
import { loadSample, uploadFile, validateFile } from "../api";

const FEATURES = [
  { icon: BarChart3, title: "Clear analytics", desc: "Turn raw learning data into useful trends." },
  { icon: Brain, title: "Model insights", desc: "Compare predictive models with confidence." },
  { icon: AlertTriangle, title: "Early signals", desc: "Surface students who may need support." },
  { icon: GraduationCap, title: "Course matching", desc: "Guide learners toward the right next step." },
];

export default function UploadPage({ onSession }) {
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [validation, setValidation] = useState(null);
  const inputRef = useRef();
  const navigate = useNavigate();
  const hasRequiredMappings = validation && (validation.missing_columns || []).every(({ sheet, column }) => validation.mappings?.[sheet]?.[column]);

  const completeUpload = (data) => {
    setSuccess(data);
    onSession(data.session_id);
    setTimeout(() => navigate("/dashboard"), 1100);
  };

  const handleFile = async (file) => {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".xlsx")) {
      setError("Please choose an Excel workbook in .xlsx format.");
      return;
    }
    setError(null);
    setSuccess(null);
    setSelectedFile(file);
    setLoading(true);
    try {
      const { data } = await validateFile(file);
      setValidation(data);
    } catch (requestError) {
      setError(requestError?.response?.data?.detail || "The upload could not be processed. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const updateMapping = (sheet, canonical, source) => {
    setValidation((current) => ({
      ...current,
      mappings: {
        ...current.mappings,
        [sheet]: { ...current.mappings[sheet], [canonical]: source || null },
      },
    }));
  };

  const confirmUpload = async () => {
    if (!selectedFile || !validation?.valid && !hasRequiredMappings) return;
    setError(null);
    setLoading(true);
    try {
      const { data } = await uploadFile(selectedFile, validation.mappings);
      completeUpload(data);
    } catch (requestError) {
      setError(requestError?.response?.data?.detail || "The workbook could not be loaded.");
    } finally {
      setLoading(false);
    }
  };

  const handleLoadSample = async () => {
    setError(null);
    setLoading(true);
    try {
      const { data } = await loadSample();
      completeUpload(data);
    } catch (requestError) {
      setError(requestError?.response?.data?.detail || "The sample dataset could not be loaded. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const onDrop = (event) => {
    event.preventDefault();
    setDragging(false);
    handleFile(event.dataTransfer.files[0]);
  };

  return (
    <div className="upload-page page-enter">
      <div className="upload-page-inner">
        <header className="upload-header">
          <div className="upload-brand-mark"><GraduationCap size={25} color="#fff" strokeWidth={2.5} /></div>
          <div>
            <span className="upload-kicker">Academic intelligence workspace</span>
            <div className="upload-brand">Student Course Analyzer</div>
          </div>
          <div className="upload-header-status"><span /> Ready for your dataset</div>
        </header>

        <div className="upload-layout">
          <section className="upload-intro">
            <span className="upload-eyebrow"><Sparkles size={13} /> From workbook to insight</span>
            <h1>See where learning is thriving. <em>Act early</em> where it is not.</h1>
            <p>Bring your course data into one clear workspace for performance analytics, predictive models, early intervention, and personalized pathways.</p>
            <div className="upload-proof-row">
              <div><strong>05</strong><span>decision views</span></div>
              <div><strong>04</strong><span>ML baselines</span></div>
              <div><strong>01</strong><span>shared dataset</span></div>
            </div>
          </section>

          <section className="upload-panel" aria-labelledby="upload-title">
            <div className="upload-panel-heading">
              <div>
                <span className="upload-panel-index">01</span>
                <div><h2 id="upload-title">Start with your workbook</h2><p>Upload an Excel file to create a private analysis session.</p></div>
              </div>
              <span className="upload-format">.XLSX</span>
            </div>

            <div
              className={`upload-zone${dragging ? " drag-over" : ""}`}
              onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              onClick={() => inputRef.current?.click()}
              role="button"
              tabIndex="0"
              onKeyDown={(event) => event.key === "Enter" && inputRef.current?.click()}
            >
              <input ref={inputRef} type="file" accept=".xlsx" hidden onChange={(event) => handleFile(event.target.files[0])} />
              <div className="upload-icon">
                {loading ? <div className="spinner" style={{ width: 25, height: 25 }} /> : <FileSpreadsheet size={27} />}
              </div>
              <div className="upload-title">{loading ? "Preparing your analysis..." : dragging ? "Release to upload" : "Drop your workbook here"}</div>
              <p className="upload-subtitle" style={{ marginTop: 6 }}>{loading ? "Validating sheets and calculating metrics" : "or select a file from your computer"}</p>
              <p className="upload-schema-note">Expected sheets: Courses, Students, Enrollments, Feedback, Platform_Performance</p>
            </div>

            <div className="quick-start">
              <div className="quick-start-divider"><span>OR QUICK START</span></div>
              <button type="button" className="btn btn-ghost" onClick={handleLoadSample} disabled={loading}>
                <Sparkles size={15} /> Load the sample dataset <ArrowRight size={14} />
              </button>
            </div>

            {error && <div className="insight-card insight-warning" style={{ marginTop: 15 }}><AlertTriangle size={16} /> {error}</div>}
            {success && (
              <div className="insight-card insight-success" style={{ marginTop: 15 }}>
                <CheckCircle size={16} />
                <span><strong>Analysis ready.</strong> Loaded {success.rows.toLocaleString()} rows across {success.students} students and {success.courses} courses.</span>
              </div>
            )}

            {validation && (
              <div className="validation-panel">
                <div className="validation-heading">
                  <div><span className={`validation-icon ${validation.valid || hasRequiredMappings ? "is-good" : "is-error"}`}>{validation.valid || hasRequiredMappings ? <CheckCircle size={16} /> : <AlertTriangle size={16} />}</span><div><h3>Preflight review</h3><p>{validation.valid || hasRequiredMappings ? "Your workbook can be analyzed. Review the warnings and mappings below." : "Fix the required issues before analysis can begin."}</p></div></div>
                  <span className={`validation-status ${validation.valid || hasRequiredMappings ? "is-good" : "is-error"}`}>{validation.valid || hasRequiredMappings ? "Ready" : "Needs attention"}</span>
                </div>

                <div className="validation-stats">
                  <div><strong>{validation.sheets?.length || 0}</strong><span>Sheets</span></div>
                  <div><strong>{validation.summary?.rows?.toLocaleString() || 0}</strong><span>Enrollment rows</span></div>
                  <div><strong>{validation.summary?.warnings || 0}</strong><span>Warnings</span></div>
                </div>

                {validation.errors?.length > 0 && <div className="validation-list is-error"><strong>Blocking issues</strong>{validation.errors.map((item) => <span key={item}>• {item}</span>)}</div>}
                {(validation.invalid_values?.length > 0 || validation.duplicates?.length > 0 || validation.inconsistent_ids?.length > 0) && (
                  <div className="validation-list is-warning">
                    <strong>Data quality warnings</strong>
                    {validation.invalid_values?.map((item) => <span key={`${item.sheet}-${item.column}`}>• {item.sheet}.{item.column}: {item.count} invalid values</span>)}
                    {validation.duplicates?.map((item) => <span key={item.sheet}>• {item.sheet}: {item.count} duplicate rows</span>)}
                    {validation.inconsistent_ids?.map((item) => <span key={item.relationship}>• {item.relationship}: {item.count} unmatched IDs</span>)}
                  </div>
                )}

                <details className="mapping-details" open={validation.missing_columns?.length > 0}>
                  <summary>Review column mapping</summary>
                  <p>Map source headers to the names used by the analytics engine. Unneeded optional columns can remain blank.</p>
                  {Object.entries(validation.mappings || {}).map(([sheet, fields]) => {
                    const columns = validation.sheets?.find((item) => item.name === sheet)?.columns || [];
                    return <div className="mapping-group" key={sheet}><h4>{sheet}</h4>{Object.entries(fields).map(([canonical, source]) => (
                      <label className="mapping-row" key={`${sheet}-${canonical}`}><span>{canonical}{["Student_ID", "Course_ID", "Score"].includes(canonical) && <b>*</b>}</span><select value={source || ""} onChange={(event) => updateMapping(sheet, canonical, event.target.value)}><option value="">Not mapped</option>{columns.map((column) => <option key={column} value={column}>{column}</option>)}</select></label>
                    ))}</div>;
                  })}
                </details>

                <button type="button" className="btn btn-primary validation-confirm" onClick={confirmUpload} disabled={(!validation.valid && !hasRequiredMappings) || loading}>
                  {loading ? "Creating analysis..." : "Confirm and analyze"}<ArrowRight size={15} />
                </button>
              </div>
            )}
          </section>
        </div>

        <section className="upload-feature-grid" aria-label="Platform capabilities">
          {FEATURES.map(({ icon: Icon, title, desc }) => (
            <article key={title} className="upload-feature">
              <span className="upload-feature-icon"><Icon size={17} /></span>
              <div><h3 className="upload-feature-title">{title}</h3><p className="upload-feature-desc">{desc}</p></div>
            </article>
          ))}
        </section>
      </div>
    </div>
  );
}
