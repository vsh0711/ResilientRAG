import { HealingStep } from "../lib/api";

export default function HealingTrace({ steps }: { steps: HealingStep[] }) {
  if (steps.length === 0) {
    return (
      <div className="trace trace-empty">
        <span className="dot dot-ok" /> No healing needed — passed on the first attempt.
      </div>
    );
  }

  return (
    <div className="trace">
      <h3>Healing trace ({steps.length} {steps.length === 1 ? "retry" : "retries"})</h3>
      <ol>
        {steps.map((step) => (
          <li key={step.retry_number} className="trace-step">
            <div className="trace-step-header">
              <span className="badge">retry {step.retry_number}</span>
              <span className="badge badge-fail">{step.failure_reason}</span>
            </div>
            <p>{step.action_taken}</p>
            <div className="trace-step-meta">
              <span>{step.previous_retrieval_mode} → {step.new_retrieval_mode}</span>
              <span>budget {step.previous_budget} → {step.new_budget}</span>
            </div>
            {step.query_rewritten && (
              <p className="rewritten-query">rewritten query: “{step.rewritten_query}”</p>
            )}
          </li>
        ))}
      </ol>
    </div>
  );
}
