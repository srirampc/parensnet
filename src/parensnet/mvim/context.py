import numpy as np

from ..types import NDFloatArray
#  function get_weight(::PIDCNetworkInference, i, j, scores, weights, nodes)
#      score = scores[i, j]
#      scores_i = vcat(scores[1:i-1, i], scores[i+1:end, i])
#      scores_j = vcat(scores[1:j-1, j], scores[j+1:end, j])
#      try
#          weights[i, j] = cdf(fit(Gamma, scores_i), score) + cdf(fit(Gamma, scores_j), score)
#      catch
#          println(string("Gamma distribution failed for ", nodes[i].label, " and ", nodes[j].label, "; used normal instead."))
#          apply_clr_context(i, j, score, scores_i, scores_j, weights)
#      end
#  end


# TODO:: pidc weight fit gamma distribution
# def get_pidc_weight(puc_scores: NDFloatArray, i:int, j: int):
#     score = puc_scores[(i, j)]
#     score_i = np.concat([puc_scores[:i, i], puc_scores[(i+1):, i]])
#     score_j = np.concat([puc_scores[:j, j], puc_scores[(j+1):, j]])
#     # TODO:: fit gamma distribution
#     pass


#    function get_weight(::CLRNetworkInference, i, j, scores, weights, nodes)
#        score = scores[i, j]
#        scores_i = vcat(scores[1:i-1, i], scores[i+1:end, i])
#        scores_j = vcat(scores[1:j-1, j], scores[j+1:end, j])
#        apply_clr_context(i, j, score, scores_i, scores_j, weights)
#    end
#
#    function apply_clr_context(i, j, score, scores_i, scores_j, weights)
#        difference_i = score - mean(scores_i)
#        difference_j = score - mean(scores_j)
#        weights[i, j] = sqrt(
#            (var(scores_i) == 0 || difference_i < 0 ? 0 : difference_i^2 / var(scores_i)) +
#            (var(scores_j) == 0 || difference_j < 0 ? 0 : difference_j^2 / var(scores_j))
#        )
#    end

def get_clr_weight(puc_scores: NDFloatArray, i: int, j: int):
    score = puc_scores[(i, j)]
    scores_i = np.concat([puc_scores[:i, i], puc_scores[(i+1):, i]])
    scores_j = np.concat([puc_scores[:j, j], puc_scores[(j+1):, j]])
    diff_i = score - np.mean(scores_i)
    diff_j = score - np.mean(scores_j)
    var_i = np.var(scores_i)
    var_j = np.var(scores_j)
    return np.sqrt((
        0.0 if (var_i == 0 or diff_i < 0) else (np.square(diff_i) / var_i)
    ) + (
        0.0 if (var_j == 0 or diff_j < 0) else (np.square(diff_j) / var_j)
    ))

