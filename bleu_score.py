import math
from collections import Counter

def extract_n_gram(n_gram: int, input_text: str):
  words = input_text.split(' ')
  result = []
  for iter in range(len(words) - n_gram + 1):
    result.append(' '.join(words[iter:iter+n_gram]))
  return result

def compute_precision(i_gram: int, candidate: str, references: list[str]):
  candidate_n_gram = extract_n_gram(i_gram, candidate)
  reference_n_grams = [extract_n_gram(i_gram, ref) for ref in references]
  candidate_counter = Counter(candidate_n_gram)
  reference_counters = [Counter(ref_ngrams) for ref_ngrams in reference_n_grams]

  match_counts = Counter()
  for n_gram in candidate_counter:
    max_count_in_refs = max(ref_counter[n_gram] for ref_counter in reference_counters)
    match_counts[n_gram] = min(candidate_counter[n_gram], max_count_in_refs)

  total_matches = sum(match_counts.values())
  total_n_grams = len(candidate_n_gram)

  return total_matches / total_n_grams

def compute_bleu_score(candidate: str, references: list[str]):
  output_length = len(candidate.split(' '))
  reference_lengths = [len(ref.split(' ')) for ref in references]
  reference_length = min(reference_lengths, key=lambda x: abs(x - output_length))

  brevity_penalty = min(1, math.exp(1 - reference_length / output_length)) # accounts for the short length candidate
  ps_n = []
  for i in range(1, 5):
    precision_i = compute_precision(i, candidate, references)
    ps_n.append(precision_i)

  bleu_score = brevity_penalty * math.exp(sum([1/4 * math.log(p) for p in ps_n]))

  return bleu_score

if __name__ == "__main__":
  candidate1 = "The cat is on the mat"
  references1 = ["The cat is on the mat", "A cat sits on a mat"]

  candidate2 = "Machine learning is a branch of artificial intelligence"
  references2 = ["Machine learning is a field of artificial intelligence", "Artificial intelligence encompasses machine learning"]

  print(compute_bleu_score(candidate1, references1))
  print(compute_bleu_score(candidate2, references2))
