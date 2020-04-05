#Copied Code
import rouge
from rouge.example import metric

class RougeMetric:

  def prepare_results(self,p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)

  def computeRouge(self, all_references, all_hypothesis):
    for aggregator in ['Avg', 'Best', 'Individual']:
      print('Evaluation with {}'.format(aggregator))
      apply_avg = aggregator == 'Avg'
      apply_best = aggregator == 'Best'

      evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                              max_n=4,
                              limit_length=True,
                              length_limit=100,
                              length_limit_type='words',
                              apply_avg=apply_avg,
                              apply_best=apply_best,
                              alpha=0.5, # Default F1_score
                              weight_factor=1.2,
                              stemming=True)


      scores = evaluator.get_scores(all_hypothesis, all_references)

      for metric, results in sorted(scores.items(), key=lambda x: x[0]):
        if not apply_avg and not apply_best: # value is a type of list as we evaluate each summary vs each reference
          for hypothesis_id, results_per_ref in enumerate(results):
           nb_references = len(results_per_ref['p'])
           for reference_id in range(nb_references):
              print('\tHypothesis #{} & Reference #{}: '.format(hypothesis_id, reference_id))
              print('\t' + self.prepare_results(results_per_ref['p'][reference_id], results_per_ref['r'][reference_id], results_per_ref['f'][reference_id]))
          print()
        else:
          print(self.prepare_results(results['p'], results['r'], results['f']))
      print()

# References
# [1] Py-rouge, computer code, downloaded 31 March 2020,
# <https://pypi.org/project/py-rouge/>.