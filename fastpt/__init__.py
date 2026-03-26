'''
	FASTPT is a numerical algorithm to calculate
	1-loop contributions to the matter power spectrum
	and other integrals of a similar type.
	The method is presented in papers arXiv:1603.04826 and arXiv:1609.05978
	Please cite these papers if you are using FASTPT in your research.

	Joseph E. McEwen (c) 2016
	mcewen.24@osu.edu

	Xiao Fang
	fang.307@osu.edu

	Jonathan A. Blazek
	blazek@berkeley.edu


	FFFFFFFF    A           SSSSSSSSS   TTTTTTTTTTTTTT             PPPPPPPPP    TTTTTTTTTTTT
	FF     	   A A         SS                 TT                   PP      PP        TT
	FF        A   A        SS                 TT                   PP      PP        TT
	FFFFF    AAAAAAA        SSSSSSSS          TT       ==========  PPPPPPPPP         TT
	FF      AA     AA              SS         TT                   PP                TT
	FF     AA       AA             SS         TT                   PP                TT
	FF    AA         AA    SSSSSSSSS          TT                   PP                TT


	The FASTPT class is the workhorse of the FASTPT algorithm.
	This class calculates integrals of the form:

	\int \frac{d^3q}{(2 \pi)^3} K(q,k-q) P(q) P(|k-q|)

	\int \frac{d^3q_1}{(2 \pi)^3} K(\hat{q_1} \dot \hat{q_2},\hat{q_1} \dot \hat{k}, \hat{q_2} \dot \hat{k}, q_1, q_2) P(q_1) P(|k-q_1|)

'''

from .info import __version__
# Core functionality
from .core.FASTPT import FASTPT
from .core.FASTPT_simple import FASTPT as FASTPT_simple
from .core.FPTHandler import FPTHandler
from .core.CacheManager import CacheManager


from .misc import HT
