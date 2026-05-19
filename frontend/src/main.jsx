import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactMarkdown from "react-markdown";
import {
  AlertTriangle,
  Bot,
  CheckCircle2,
  Clock3,
  Map,
  MemoryStick,
  PlaneTakeoff,
  RefreshCw,
  Route,
  Send,
  Sparkles,
  Star,
  WalletCards
} from "lucide-react";
import "./styles.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8010";

const sampleRequests = [
  "去上海1天，预算500元，喜欢博物馆，不要太赶",
  "去杭州2天，预算900元，喜欢自然、博物馆和本地小吃，轻松一点",
  "去北京2天，预算1200元，喜欢历史和胡同，不想购物"
];

function App() {
  const [text, setText] = useState(sampleRequests[0]);
  const [userId, setUserId] = useState("demo");
  const [mock, setMock] = useState(false);
  const [reviewOnly, setReviewOnly] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [selectedVariantId, setSelectedVariantId] = useState("");

  async function submitPlan(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/trips/plan`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          user_id: userId,
          auto_confirm: !reviewOnly,
          mock
        })
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "TripMind request failed");
      }
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const run = result?.run;
  const itinerary = run?.itinerary;
  const budget = itinerary?.budget;
  const issues = run?.critique?.issues || [];
  const trace = run?.trace || [];
  const artifacts = run?.artifacts || {};
  const metrics = artifacts.evaluation_metrics || {};
  const dataSources = artifacts.data_sources || [];
  const dataWarnings = artifacts.data_warnings || [];
  const clarificationQuestions = run?.intent?.clarification_questions || [];
  const blockingMissingFields = run?.intent?.blocking_missing_fields || [];
  const branchableMissingFields = run?.intent?.branchable_missing_fields || [];
  const variants = artifacts.plan_variants || [];
  const episodicHits = artifacts.episodic_memory_hits || [];
  const recommendedVariant = variants.find((variant) => variant.recommended) || null;
  const selectedVariant =
    variants.find((variant) => variant.variant_id === selectedVariantId) || recommendedVariant || null;
  const displayedItinerary = selectedVariant?.itinerary || itinerary || null;
  const displayedCritique = selectedVariant?.critique || run?.critique || null;
  const displayedBudget = displayedItinerary?.budget || budget || null;
  const displayedIssues = displayedCritique?.issues || issues;
  const displayedMetrics = displayedCritique?.metrics || metrics;
  const totalElapsed = useMemo(
    () => trace.reduce((sum, step) => sum + (Number(step.metadata?.elapsed_ms) || 0), 0),
    [trace]
  );

  useEffect(() => {
    if (recommendedVariant) {
      setSelectedVariantId(recommendedVariant.variant_id);
      return;
    }
    setSelectedVariantId("");
  }, [result, recommendedVariant]);

  return (
    <main className="app-shell">
      <section className="workspace">
        <div className="planner-pane">
          <header className="brand-row">
            <div className="brand-mark">
              <PlaneTakeoff size={24} />
            </div>
            <div>
              <h1>TripMind</h1>
              <p>Multi-agent travel planning with MCP travel tools</p>
            </div>
          </header>

          <form className="request-panel" onSubmit={submitPlan}>
            <label htmlFor="trip-request">Travel request</label>
            <textarea
              id="trip-request"
              value={text}
              onChange={(event) => setText(event.target.value)}
              rows={7}
            />

            <div className="sample-row">
              {sampleRequests.map((sample) => (
                <button key={sample} type="button" onClick={() => setText(sample)}>
                  {sample.slice(0, 10)}...
                </button>
              ))}
            </div>

            <div className="control-grid">
              <label>
                <span>User ID</span>
                <input value={userId} onChange={(event) => setUserId(event.target.value)} />
              </label>

              <label className="toggle-line">
                <input type="checkbox" checked={mock} onChange={(event) => setMock(event.target.checked)} />
                <span>Skip LLM</span>
              </label>

              <label className="toggle-line">
                <input type="checkbox" checked={reviewOnly} onChange={(event) => setReviewOnly(event.target.checked)} />
                <span>Review only</span>
              </label>
            </div>

            <button className="primary-action" type="submit" disabled={loading || !text.trim()}>
              {loading ? <RefreshCw className="spin" size={18} /> : <Send size={18} />}
              {loading ? "Planning..." : "Plan trip"}
            </button>
          </form>

          {error && (
            <div className="error-box">
              <AlertTriangle size={18} />
              <span>{error}</span>
            </div>
          )}
        </div>

        <div className="result-pane">
          {!result && !loading && <EmptyState />}
          {loading && <LoadingState />}
          {result && (
            <>
              <section className="metric-grid">
                <Metric icon={<CheckCircle2 />} label="State" value={run.state} />
                <Metric icon={<WalletCards />} label="Budget" value={displayedBudget ? `${Math.round(displayedBudget.total)} ${displayedBudget.currency}` : "N/A"} />
                <Metric icon={<AlertTriangle />} label="Issues" value={displayedIssues.length} />
                <Metric icon={<Clock3 />} label="Agent time" value={`${(totalElapsed / 1000).toFixed(1)}s`} />
              </section>

              {blockingMissingFields.length > 0 && (
                <section className="review-section">
                  <div className="section-heading">
                    <AlertTriangle size={18} />
                    <h2>Need clarification</h2>
                  </div>
                  <div className="issue-list">
                    {clarificationQuestions.map((question) => (
                      <div className="issue-card warning" key={question}>
                        <strong>question</strong>
                        <p>{question}</p>
                      </div>
                    ))}
                  </div>
                </section>
              )}

              {branchableMissingFields.length > 0 && variants.length > 0 && (
                <section className="review-section">
                  <div className="section-heading">
                    <WalletCards size={18} />
                    <h2>Budget variants</h2>
                  </div>
                  <div className="variant-grid">
                    {variants.map((variant) => (
                      <button
                        type="button"
                        className={`variant-card ${selectedVariantId === variant.variant_id ? "active" : ""}`}
                        key={variant.variant_id}
                        onClick={() => setSelectedVariantId(variant.variant_id)}
                      >
                        <strong>
                          {variant.label}
                          {variant.recommended ? " · recommended" : ""}
                        </strong>
                        <p>
                          ~{Math.round(variant.itinerary.budget.total)} {variant.itinerary.budget.currency} ·
                          {" "}issues {variant.critique.issues.length}
                        </p>
                        <span>{variant.reason}</span>
                      </button>
                    ))}
                  </div>
                </section>
              )}

              <section className="source-panel">
                <div className="section-heading compact">
                  <Star size={16} />
                  <h2>Data sources</h2>
                </div>
                <div className="source-chip-row">
                  {dataSources.length ? (
                    dataSources.map((source) => (
                      <span className="source-chip" key={source}>
                        {humanizeSource(source)}
                      </span>
                    ))
                  ) : (
                    <span className="muted">No source metadata</span>
                  )}
                </div>
                {dataWarnings.length > 0 && (
                  <div className="source-warning-list">
                    {dataWarnings.map((warning) => (
                      <div className="source-warning" key={warning}>
                        <AlertTriangle size={14} />
                        <span>{warning}</span>
                      </div>
                    ))}
                  </div>
                )}
              </section>

              <section className="trace-strip">
                {trace.map((step) => (
                  <div className="trace-step" key={`${step.agent}-${step.state}`}>
                    <Bot size={16} />
                    <strong>{step.agent}</strong>
                    <span>{step.metadata?.elapsed_ms ? `${step.metadata.elapsed_ms}ms` : step.state}</span>
                  </div>
                ))}
              </section>

              {displayedItinerary && (
                <section className="days-section">
                  <div className="section-heading">
                    <Map size={18} />
                    <h2>{selectedVariant ? `${selectedVariant.label} preview` : "Daily plan"}</h2>
                  </div>
                  {displayedItinerary.days.map((day) => (
                    <article className="day-card" key={day.day}>
                      <div className="day-title">
                        <span>Day {day.day}</span>
                        <strong>{day.theme}</strong>
                      </div>
                      <div className="timeline">
                        {day.items.map((item, index) => (
                          <div className="timeline-item" key={`${item.title}-${index}`}>
                            <span className="time-pill">{displayTimeOfDay(item.time_of_day)}</span>
                            <div>
                              <strong>{item.title}</strong>
                              <p>{item.area} · {item.duration_minutes} min · ~{Math.round(item.estimated_cost)} {displayedBudget?.currency || "CNY"}</p>
                              <div className="item-meta-row">
                                {item.source && <span className="meta-chip">{humanizeSource(item.source)}</span>}
                                {typeof item.rating === "number" && item.rating > 0 && (
                                  <span className="meta-chip">
                                    <Star size={12} />
                                    {item.rating.toFixed(1)}
                                  </span>
                                )}
                                {typeof item.review_count === "number" && item.review_count > 0 && (
                                  <span className="meta-chip">{item.review_count} reviews</span>
                                )}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                      <div className="transit-line">
                        <Route size={16} />
                        {day.transit.length
                          ? day.transit.map((t) => `${t.origin}→${t.destination} ${t.minutes}min`).join(" · ")
                          : "No local transit estimate"}
                      </div>
                    </article>
                  ))}
                </section>
              )}

              <section className="review-section">
                <div className="section-heading">
                  <Sparkles size={18} />
                  <h2>{selectedVariant ? `${selectedVariant.label} review` : "Critic review"}</h2>
                </div>
                {Object.keys(displayedMetrics).length > 0 && (
                  <div className="metrics-table">
                    {Object.entries(displayedMetrics).map(([key, value]) => (
                      <div key={key}>
                        <span>{key}</span>
                        <strong>{value}</strong>
                      </div>
                    ))}
                  </div>
                )}
                {displayedIssues.length === 0 ? (
                  <p className="muted">No major issues found.</p>
                ) : (
                  <div className="issue-list">
                    {displayedIssues.map((issue) => (
                      <div className={`issue-card ${issue.severity}`} key={`${issue.code}-${issue.message}`}>
                        <strong>{issue.code}</strong>
                        <p>{issue.message}</p>
                        <span>{issue.suggestion}</span>
                      </div>
                    ))}
                  </div>
                )}
              </section>

              <section className="markdown-section">
                <div className="section-heading">
                  <MemoryStick size={18} />
                  <h2>Artifacts and Markdown</h2>
                </div>
                <div className="artifact-list">
                  <div><strong>Run</strong><span>{run.run_id || "N/A"}</span></div>
                  <div><strong>Attractions</strong><span>{(artifacts.selected_attractions || []).join(", ") || "N/A"}</span></div>
                  <div><strong>Restaurants</strong><span>{(artifacts.selected_restaurants || []).join(", ") || "N/A"}</span></div>
                  <div><strong>Replans</strong><span>{artifacts.replan_count || 0}</span></div>
                </div>
                {episodicHits.length > 0 && (
                  <div className="artifact-list">
                    <div><strong>Episodic memory</strong><span>{episodicHits.map((entry) => entry.summary).join(" | ")}</span></div>
                  </div>
                )}
                <div className="markdown-body">
                  <ReactMarkdown>{result.markdown}</ReactMarkdown>
                </div>
              </section>
            </>
          )}
        </div>
      </section>
    </main>
  );
}

function Metric({ icon, label, value }) {
  return (
    <div className="metric-card">
      {React.cloneElement(icon, { size: 18 })}
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function humanizeSource(source) {
  const labels = {
    baidu_place: "Baidu Place API",
    amap_place: "Amap Place API",
    osm_overpass: "OSM Overpass",
    osm_nominatim: "OSM Nominatim"
  };
  return labels[source] || source;
}

function displayTimeOfDay(value) {
  const labels = {
    Breakfast: "早餐",
    Morning: "上午行程",
    Lunch: "中餐",
    Afternoon: "下午行程",
    "Late afternoon": "傍晚前行程",
    Dinner: "晚餐",
    Evening: "晚上行程"
  };
  return labels[value] || value;
}

function EmptyState() {
  return (
    <div className="empty-state">
      <Sparkles size={30} />
      <h2>Ready for a real agent run</h2>
      <p>Submit a request to exercise IntentAgent, PlannerAgent, CriticAgent, Memory, and MCP travel tools.</p>
    </div>
  );
}

function LoadingState() {
  return (
    <div className="empty-state">
      <RefreshCw className="spin" size={30} />
      <h2>Agents are working</h2>
      <p>TripMind is calling the LLM and MCP travel tools. Real POI queries may take a few seconds.</p>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
