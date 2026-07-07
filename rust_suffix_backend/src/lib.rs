use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass]
struct RustSuffixCache {
    max_tree_depth: usize,
    max_spec_factor: f64,
    min_token_prob: f64,
    min_match_tokens: usize,
    next_counts: HashMap<Vec<i64>, HashMap<i64, u32>>,
}

#[pymethods]
impl RustSuffixCache {
    #[new]
    fn new(
        max_tree_depth: usize,
        _max_cached_requests: usize,
        max_spec_factor: f64,
        min_token_prob: f64,
        min_match_tokens: usize,
    ) -> PyResult<Self> {
        if max_tree_depth < 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "max_tree_depth must be >= 1",
            ));
        }
        if max_spec_factor <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "max_spec_factor must be > 0",
            ));
        }
        if !(0.0..=1.0).contains(&min_token_prob) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "min_token_prob must be in [0, 1]",
            ));
        }
        if min_match_tokens < 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "min_match_tokens must be >= 1",
            ));
        }
        Ok(Self {
            max_tree_depth,
            max_spec_factor,
            min_token_prob,
            min_match_tokens,
            next_counts: HashMap::new(),
        })
    }

    fn add_sequence(&mut self, token_ids: Vec<i64>) {
        self.add_sequence_with_min_next_index(token_ids, 1);
    }

    fn add_sequence_with_min_next_index(&mut self, token_ids: Vec<i64>, min_next_index: usize) {
        if token_ids.len() < 2 {
            return;
        }
        if self.max_tree_depth < self.min_match_tokens {
            return;
        }
        let start_next_index = min_next_index.max(1).min(token_ids.len());
        for next_index in start_next_index..token_ids.len() {
            let next_token = token_ids[next_index];
            let start = next_index.saturating_sub(self.max_tree_depth);
            if next_index < self.min_match_tokens {
                continue;
            }
            let stop = next_index - self.min_match_tokens + 1;
            if stop <= start {
                continue;
            }
            for suffix_start in start..stop {
                let suffix = token_ids[suffix_start..next_index].to_vec();
                let counts = self.next_counts.entry(suffix).or_default();
                *counts.entry(next_token).or_insert(0) += 1;
            }
        }
    }

    fn remove_sequence(&mut self, token_ids: Vec<i64>) {
        self.remove_sequence_with_min_next_index(token_ids, 1);
    }

    fn remove_sequence_with_min_next_index(&mut self, token_ids: Vec<i64>, min_next_index: usize) {
        if token_ids.len() < 2 {
            return;
        }
        if self.max_tree_depth < self.min_match_tokens {
            return;
        }
        let mut empty_suffixes: Vec<Vec<i64>> = Vec::new();
        let start_next_index = min_next_index.max(1).min(token_ids.len());
        for next_index in start_next_index..token_ids.len() {
            let next_token = token_ids[next_index];
            let start = next_index.saturating_sub(self.max_tree_depth);
            if next_index < self.min_match_tokens {
                continue;
            }
            let stop = next_index - self.min_match_tokens + 1;
            if stop <= start {
                continue;
            }
            for suffix_start in start..stop {
                let suffix = token_ids[suffix_start..next_index].to_vec();
                let Some(counts) = self.next_counts.get_mut(&suffix) else {
                    continue;
                };
                if let Some(count) = counts.get_mut(&next_token) {
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        counts.remove(&next_token);
                    }
                }
                if counts.is_empty() {
                    empty_suffixes.push(suffix);
                }
            }
        }
        for suffix in empty_suffixes {
            self.next_counts.remove(&suffix);
        }
    }

    fn propose(&self, context_token_ids: Vec<i64>, max_tokens: usize) -> Vec<i64> {
        self.propose_from_slice(&context_token_ids, max_tokens)
    }

    fn propose_many(&self, context_batches: Vec<Vec<i64>>, max_tokens: usize) -> Vec<Vec<i64>> {
        context_batches
            .iter()
            .map(|context_token_ids| self.propose_from_slice(context_token_ids, max_tokens))
            .collect()
    }

    fn propose_many_with_adapter_inputs(
        &self,
        context_batches: Vec<Vec<i64>>,
        suffix_max_tokens: usize,
        max_adapter_tokens: usize,
    ) -> (Vec<Vec<i64>>, Vec<Vec<i64>>, Vec<usize>) {
        let mut suffix_batches: Vec<Vec<i64>> = Vec::with_capacity(context_batches.len());
        let mut adapter_contexts: Vec<Vec<i64>> = Vec::with_capacity(context_batches.len());
        let mut adapter_budgets: Vec<usize> = Vec::with_capacity(context_batches.len());

        for context in context_batches.iter() {
            let suffix_tokens = self.propose_from_slice(context, suffix_max_tokens);
            let adapter_budget = if context.is_empty() {
                0
            } else {
                max_adapter_tokens
            };
            let mut adapter_context = Vec::with_capacity(context.len() + suffix_tokens.len());
            adapter_context.extend_from_slice(context);
            adapter_context.extend_from_slice(&suffix_tokens);

            suffix_batches.push(suffix_tokens);
            adapter_contexts.push(adapter_context);
            adapter_budgets.push(adapter_budget);
        }

        (suffix_batches, adapter_contexts, adapter_budgets)
    }

    fn suffix_count(&self) -> usize {
        self.next_counts.len()
    }

    fn clear(&mut self) {
        self.next_counts.clear();
    }
}

impl RustSuffixCache {
    fn propose_from_slice(&self, context: &[i64], max_tokens: usize) -> Vec<i64> {
        if max_tokens == 0 || context.is_empty() {
            return Vec::new();
        }
        let Some(initial_match_len) = self.longest_suffix_match_len(context) else {
            return Vec::new();
        };
        let max_by_factor = ((initial_match_len as f64) * self.max_spec_factor)
            .floor()
            .max(1.0) as usize;
        let budget = max_tokens.min(max_by_factor);
        let mut proposed = Vec::with_capacity(budget);
        let mut rolling_context = context.to_vec();
        for _ in 0..budget {
            let Some(match_len) = self.longest_suffix_match_len(&rolling_context) else {
                break;
            };
            let suffix_start = rolling_context.len() - match_len;
            let suffix = &rolling_context[suffix_start..];
            let Some(next_token) = self.best_next_token(suffix) else {
                break;
            };
            proposed.push(next_token);
            rolling_context.push(next_token);
        }
        proposed
    }

    fn longest_suffix_match_len(&self, context: &[i64]) -> Option<usize> {
        let max_depth = context.len().min(self.max_tree_depth);
        let min_depth = self.min_match_tokens;
        if max_depth < min_depth {
            return None;
        }
        for size in (min_depth..=max_depth).rev() {
            let suffix = &context[context.len() - size..];
            if self.next_counts.contains_key(suffix) {
                return Some(size);
            }
        }
        None
    }

    fn best_next_token(&self, suffix: &[i64]) -> Option<i64> {
        let counts = self.next_counts.get(suffix)?;
        let total: u32 = counts.values().copied().sum();
        if total == 0 {
            return None;
        }
        let (token_id, count) = counts
            .iter()
            .max_by(|(left_token, left_count), (right_token, right_count)| {
                left_count
                    .cmp(right_count)
                    .then_with(|| right_token.cmp(left_token))
            })?;
        if (*count as f64) / (total as f64) < self.min_token_prob {
            return None;
        }
        Some(*token_id)
    }
}

#[pyfunction]
fn merge_hybrid_drafts(
    suffix_batches: Vec<Vec<i64>>,
    jetspec_batches: Vec<Vec<i64>>,
    num_speculative_tokens: usize,
) -> (Vec<Vec<i64>>, Vec<usize>, Vec<usize>) {
    let batch_count = suffix_batches.len().max(jetspec_batches.len());
    let mut proposals: Vec<Vec<i64>> = Vec::with_capacity(batch_count);
    let mut suffix_counts: Vec<usize> = Vec::with_capacity(batch_count);
    let mut jetspec_counts: Vec<usize> = Vec::with_capacity(batch_count);

    for index in 0..batch_count {
        let suffix_tokens = suffix_batches.get(index).map(Vec::as_slice).unwrap_or(&[]);
        let jetspec_tokens = jetspec_batches.get(index).map(Vec::as_slice).unwrap_or(&[]);
        let suffix_take = suffix_tokens.len().min(num_speculative_tokens);
        let jetspec_take = jetspec_tokens
            .len()
            .min(num_speculative_tokens.saturating_sub(suffix_take));

        let mut proposal = Vec::with_capacity(suffix_take + jetspec_take);
        proposal.extend_from_slice(&suffix_tokens[..suffix_take]);
        proposal.extend_from_slice(&jetspec_tokens[..jetspec_take]);

        suffix_counts.push(suffix_take);
        jetspec_counts.push(jetspec_take);
        proposals.push(proposal);
    }

    (proposals, suffix_counts, jetspec_counts)
}

#[pyfunction]
fn merge_and_summarize_hybrid_drafts(
    suffix_batches: Vec<Vec<i64>>,
    jetspec_batches: Vec<Vec<i64>>,
    jetspec_raw_counts: Vec<usize>,
    adapter_budget_counts: Vec<usize>,
    num_speculative_tokens: usize,
    suffix_budget_tokens: usize,
    force_hybrid_proposals: bool,
) -> (Vec<Vec<i64>>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let batch_count = suffix_batches.len().max(jetspec_batches.len());
    let mut proposals: Vec<Vec<i64>> = Vec::with_capacity(batch_count);
    let mut suffix_counts: Vec<usize> = Vec::with_capacity(batch_count);
    let mut jetspec_counts: Vec<usize> = Vec::with_capacity(batch_count);

    for index in 0..batch_count {
        let suffix_tokens = suffix_batches.get(index).map(Vec::as_slice).unwrap_or(&[]);
        let jetspec_tokens = jetspec_batches.get(index).map(Vec::as_slice).unwrap_or(&[]);
        let adapter_budget = adapter_budget_counts.get(index).copied().unwrap_or(0);
        let (suffix_take, jetspec_take) =
            if force_hybrid_proposals && adapter_budget > 0 && !jetspec_tokens.is_empty() {
                let reserved_jetspec = num_speculative_tokens
                    .min(adapter_budget)
                    .min(jetspec_tokens.len());
                let suffix_take = suffix_tokens
                    .len()
                    .min(num_speculative_tokens.saturating_sub(reserved_jetspec));
                let jetspec_take = jetspec_tokens
                    .len()
                    .min(num_speculative_tokens.saturating_sub(suffix_take));
                (suffix_take, jetspec_take)
            } else {
                let suffix_take = suffix_tokens.len().min(num_speculative_tokens);
                let jetspec_take = jetspec_tokens
                    .len()
                    .min(num_speculative_tokens.saturating_sub(suffix_take));
                (suffix_take, jetspec_take)
            };

        let mut proposal = Vec::with_capacity(suffix_take + jetspec_take);
        proposal.extend_from_slice(&suffix_tokens[..suffix_take]);
        proposal.extend_from_slice(&jetspec_tokens[..jetspec_take]);

        suffix_counts.push(suffix_take);
        jetspec_counts.push(jetspec_take);
        proposals.push(proposal);
    }

    let summary = summarize_hybrid_counts_impl(
        &suffix_counts,
        &jetspec_counts,
        &jetspec_raw_counts,
        &adapter_budget_counts,
        num_speculative_tokens,
        suffix_budget_tokens,
        force_hybrid_proposals,
    );

    (proposals, suffix_counts, jetspec_counts, summary)
}

#[pyfunction]
fn filter_jetspec_drafts(
    draft_batches: Vec<Vec<i64>>,
    prob_batches: Vec<Vec<f64>>,
    budgets: Vec<usize>,
    pad_token_id: i64,
    min_token_prob: f64,
) -> Vec<Vec<i64>> {
    let mut proposals: Vec<Vec<i64>> = Vec::with_capacity(draft_batches.len());

    for (row_index, draft_row) in draft_batches.iter().enumerate() {
        let budget = budgets.get(row_index).copied().unwrap_or(0);
        let prob_row = prob_batches.get(row_index).map(Vec::as_slice);
        let mut proposal: Vec<i64> = Vec::with_capacity(budget.min(draft_row.len()));

        for (token_offset, token_id) in draft_row.iter().take(budget).enumerate() {
            if *token_id == pad_token_id {
                continue;
            }
            if min_token_prob > 0.0 {
                if let Some(probs) = prob_row {
                    if let Some(prob) = probs.get(token_offset) {
                        if *prob < min_token_prob {
                            break;
                        }
                    }
                    if token_offset >= probs.len() {
                        break;
                    }
                }
            }
            proposal.push(*token_id);
        }

        proposals.push(proposal);
    }

    proposals
}

#[pyfunction]
fn summarize_hybrid_counts(
    suffix_counts: Vec<usize>,
    jetspec_counts: Vec<usize>,
    jetspec_raw_counts: Vec<usize>,
    adapter_budget_counts: Vec<usize>,
    num_speculative_tokens: usize,
    suffix_budget_tokens: usize,
    force_hybrid_proposals: bool,
) -> Vec<usize> {
    summarize_hybrid_counts_impl(
        &suffix_counts,
        &jetspec_counts,
        &jetspec_raw_counts,
        &adapter_budget_counts,
        num_speculative_tokens,
        suffix_budget_tokens,
        force_hybrid_proposals,
    )
}

fn summarize_hybrid_counts_impl(
    suffix_counts: &[usize],
    jetspec_counts: &[usize],
    jetspec_raw_counts: &[usize],
    adapter_budget_counts: &[usize],
    num_speculative_tokens: usize,
    suffix_budget_tokens: usize,
    force_hybrid_proposals: bool,
) -> Vec<usize> {
    let sequence_count = suffix_counts.len().min(jetspec_counts.len());
    let mut proposed_tokens = 0usize;
    let mut suffix_tokens = 0usize;
    let mut jetspec_tokens = 0usize;
    let mut sequences_filled_by_suffix = 0usize;
    let mut sequences_with_suffix = 0usize;
    let mut sequences_with_jetspec = 0usize;
    let mut sequences_without_proposal = 0usize;

    for index in 0..sequence_count {
        let suffix_count = suffix_counts[index];
        let jetspec_count = jetspec_counts[index];
        let proposed_count = suffix_count + jetspec_count;

        proposed_tokens += proposed_count;
        suffix_tokens += suffix_count;
        jetspec_tokens += jetspec_count;

        if suffix_count >= num_speculative_tokens {
            sequences_filled_by_suffix += 1;
        }
        if suffix_count > 0 {
            sequences_with_suffix += 1;
        }
        if jetspec_count > 0 {
            sequences_with_jetspec += 1;
        }
        if proposed_count == 0 {
            sequences_without_proposal += 1;
        }
    }

    let jetspec_raw_tokens: usize = jetspec_raw_counts.iter().sum();
    let jetspec_unused_tokens = jetspec_raw_tokens.saturating_sub(jetspec_tokens);
    let adapter_budget_tokens: usize = adapter_budget_counts.iter().sum();
    let suffix_budget_tokens_total = sequence_count * suffix_budget_tokens;
    let suffix_attempted_sequences = if suffix_budget_tokens > 0 {
        sequence_count
    } else {
        0
    };
    let adapter_attempted_sequences = adapter_budget_counts
        .iter()
        .filter(|budget| **budget > 0)
        .count();
    let adapter_empty_sequences = adapter_budget_counts
        .iter()
        .zip(jetspec_counts.iter())
        .filter(|(budget, jetspec_count)| **budget > 0 && **jetspec_count == 0)
        .count();
    let sequences_with_jetspec_raw = jetspec_raw_counts
        .iter()
        .filter(|count| **count > 0)
        .count();
    let jetspec_filtered_tokens: usize = adapter_budget_counts
        .iter()
        .zip(jetspec_counts.iter())
        .map(|(budget, jetspec_count)| budget.saturating_sub(*jetspec_count))
        .sum();
    let forced_suffix_and_jetspec_attempts = usize::from(
        force_hybrid_proposals
            && suffix_budget_tokens > 0
            && sequence_count > 0
            && adapter_attempted_sequences == sequence_count,
    );

    vec![
        sequence_count,
        proposed_tokens,
        suffix_tokens,
        jetspec_tokens,
        jetspec_raw_tokens,
        jetspec_unused_tokens,
        adapter_budget_tokens,
        suffix_budget_tokens_total,
        suffix_attempted_sequences,
        adapter_attempted_sequences,
        adapter_empty_sequences,
        sequences_filled_by_suffix,
        sequences_with_suffix,
        sequences_with_jetspec,
        sequences_with_jetspec_raw,
        sequences_without_proposal,
        jetspec_filtered_tokens,
        forced_suffix_and_jetspec_attempts,
    ]
}

#[pymodule]
fn _vllm_dflash_jit_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustSuffixCache>()?;
    m.add_function(wrap_pyfunction!(merge_hybrid_drafts, m)?)?;
    m.add_function(wrap_pyfunction!(merge_and_summarize_hybrid_drafts, m)?)?;
    m.add_function(wrap_pyfunction!(filter_jetspec_drafts, m)?)?;
    m.add_function(wrap_pyfunction!(summarize_hybrid_counts, m)?)?;
    Ok(())
}
