from matplotlib import pyplot as plt
import alien_bayes


def two_line_plot(xvals1, yvals1, label1, xvals2, yvals2, label2, title, outfile_path):
    plt.plot(xvals1, yvals1, label=label1, color='blue', marker='.', linestyle='solid')
    plt.plot(xvals2, yvals2, label=label2, color='green', marker='.', linestyle='solid')
    plt.title(title)
    plt.legend()
    plt.savefig(outfile_path)


if __name__ == '__main__':
    nodes = alien_bayes.ALIEN_NODES
    sampler_reject = alien_bayes.RejectionSampler(nodes)
    sampler_like = alien_bayes.LikelihoodWeightingSampler(nodes)
    reject_vals = []
    like_vals = []
    x_vals = []
    lab1 = "Rejection"
    lab2 = "Likelyhood"
    title = "Rejection vs. Likelyhood"

    for x in range(100, 10000, 100):
        reject_vals.append(sampler_reject.get_prob({'A': True}, {'M': True, 'B': True}, x))
        like_vals.append(sampler_like.get_prob({'A': True}, {'M': True, 'B': True}, x))
        x_vals.append(x)

    two_line_plot(x_vals, reject_vals, lab1, x_vals, like_vals, lab2, title, "alien_bayes.pdf")




# 
# Fill in this script to empirically calculate P(A=true | M=true, B=true) using the rejection
# sampling and likelihood weighting code found in alien_bayes.py.
#
# Use the two_line_plot() function above to generate a line graph with one line for each 
# approximation technique.  The x-axis should represent different n, the number of samples 
# generated, with the probability estimate for the conditional probability above on the y-axis.  
# 
# You should generate estimates using at least 100 different values of n, and increase it to 
# the point that the estimates appear to stabilize.  Note that for rejection sampling, n should
# represent the number of simple samples created, not the number retained after rejecting those
# that do not agree with the evidence.  
# 
# Your script should produce a plot named "alien_bayes.pdf". 
#  





