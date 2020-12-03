import matplotlib.pyplot as plt

def rescale_score_by_abs (score, max_score, min_score):
	"""
	Normalize the relevance value (=score), accordingly to the extremal relevance values (max_score and min_score), 
	for visualization with a diverging colormap.
	i.e. rescale positive relevance to the range [0.5, 1.0], and negative relevance to the range [0.0, 0.5],
	using the highest absolute relevance for linear interpolation.
	"""
	
	# CASE 1: positive AND negative scores occur --------------------
	if max_score>0 and min_score<0:
	
		if max_score >= abs(min_score):   # deepest color is positive
			if score>=0:
				return 0.5 + 0.5*(score/max_score)
			else:
				return 0.5 - 0.5*(abs(score)/max_score)

		else:                             # deepest color is negative
			if score>=0:
				return 0.5 + 0.5*(score/abs(min_score))
			else:
				return 0.5 - 0.5*(score/min_score)   
	
	# CASE 2: ONLY positive scores occur -----------------------------       
	elif max_score>0 and min_score>=0: 
		if max_score == min_score:
			return 1.0
		else:
			return 0.5 + 0.5*(score/max_score)
	
	# CASE 3: ONLY negative scores occur -----------------------------
	elif max_score<=0 and min_score<0: 
		if max_score == min_score:
			return 0.0
		else:
			return 0.5 - 0.5*(score/min_score)    
  
	  
def getRGB (c_tuple):
	return "#%02x%02x%02x"%(int(c_tuple[0]*255), int(c_tuple[1]*255), int(c_tuple[2]*255))

	 
def span_word (word, score, colormap):
	return "<span style=\"background-color:"+getRGB(colormap(score))+"\">"+word+"</span>"


def html_heatmap (words, scores, cmap_name="bwr"):
	"""
	Return word-level heatmap in HTML format,
	with words being the list of words (as string),
	scores the corresponding list of word-level relevance values,
	and cmap_name the name of the matplotlib diverging colormap.
	"""
	
	colormap  = plt.get_cmap(cmap_name)
	 
	assert len(words)==len(scores)
	max_s     = max(scores)
	min_s     = min(scores)
	
	output_text = ""
	
	for idx, w in enumerate(words):
		score       = rescale_score_by_abs(scores[idx], max_s, min_s)
		output_text = output_text + span_word(w, score, colormap) + " "
	
	return output_text + "\n"



def main():
	# words = ['a', 'very', 'funny', 'movie', '.']
	# scores = [ 0.9831929,  -0.63894178,  6.53314906, -0.14404073, -0.36896541]

	# words = ['fail', 'to', 'be', 'a', 'very', 'funny', 'movie', '.']
	# scores = [7.95088891, 1.61366659, 0.75466716, 0.24075226, 2.90815666, -3.71055831, 0.70344164, -0.632309]

	# words = ['not', 'a', 'very', 'funny', 'movie', '.']
	# scores = [2.98045584, 1.17765487, 4.05663827, -4.89162298, 0.87247377, -0.63306575]

	# words = ['never', 'a', 'very', 'funny', 'movie', '.']
	# scores = [-0.24960268, 0.68699463, -0.77227701, 6.27508434, 0.14702064, -0.23515925]

	# words = ['not', 'fail', 'to', 'be', 'a', 'very', 'funny', 'movie', '.']
	# scores = [0.37775898, 8.74651897, 1.41834813, 0.42831983, -0.010791, 2.77108823, -3.6457253, 0.20855701, -0.64770563]


	# words = ['daring', ',', 'mesmerizing', 'and', 'exceedingly', 'hard', 'to', 'forget', '.']
	# scores = [3.93910044, 0.58881061, 8.99380944, 1.73294336, 2.38652828, 3.22451698, -0.60095891, -1.58178938, -0.04514259]

	# words = ['exceedingly', 'hard', 'to', 'forget']
	# scores = [0.39079047, 0.26717017, -0.04194807, 2.36610526]

	# words = ['exceedingly', 'hard', 'to', 'remember']
	# scores = [1.84871137, 2.75570145, 1.38772965, 0.46720101]

	# words = ['exceedingly', 'easy', 'to', 'forget']
	# scores = [0.33629294,  0.85781795, -0.07333658,  2.44243684]

	# words = ['exceedingly', 'easy', 'to', 'remember']
	# scores = [1.85221602, 2.59985974, 1.41942803, 0.41770288]

	# words = ['a', 'fascinating', 'and', 'fun', 'film', '.']
	# scores = [0.68951285,  5.95688863,  1.46684003,  7.77773112,  1.29891723, -0.52023544]

	# words = ['all', 'this', 'turns', 'out', 'to', 'be', 'neither', 'funny', 'nor', 'provocative', '-', 'only', 'dull', '.']
	# scores = [-0.00284927, -0.00629312, -0.0177723,  -0.00561546,  0.00187644, -0.01304687,
	# 		0.06506842, -0.07057944,  0.08516518, -0.12529564,  0.15697606,  0.13827942, 
	# 		1.27985754, -0.58936053]

	# words = ['occasionally', 'melodramatic', ',', 'it', "'s", 'also', 'extremely', 'effective', '.']
	# scores = [0.05485572,  0.01351719,  0.00878752,  0.00137596,  0.00798059,  0.02812169, 
	# 		0.03464074,  0.18600131, -0.50509051]


	# words = ['a', 'thoughtful', ',', 'moving', 'piece', 'that', 'faces', 'difficult', 'issues', 'with', 'honesty', 'and', 'beauty', '.']
	# scores = [0.07807562,  0.677239,    0.07784149,  0.26756284,  0.26323426,  0.03837715, 
	# 			0.05352856, -0.09160687, -0.10683642, -0.01862339,  0.21189614, -0.07581858,
	# 			0.37011332, -1.04398831]

	# words = ['it', "'s", 'a', 'good', 'film', '--', 'not', 'a', 'classic', ',', 'but', 'odd', ',', 'entertaining', 'and', 'authentic', '.']
	# scores = [0.57444651,  0.32416322, -0.03797868,  0.23307925,  0.08107225,  0.15773636,
	# 		-0.25387156,  0.00273272,  0.12561181,  0.03699991, -0.06325512, -0.10213069,
	# 		-0.03453131,  0.3936737,  -0.02080863,  0.32748093, -1.0317636]

	# words = ['after', 'that', ',', 'it', 'just', 'gets', 'stupid', 'and', 'maudlin', '.']
	# scores = [0.15518425, -0.53583752, -0.2574038,  -0.4045445,   0.06782995, -0.21261884,
	# 		5.09646424, -0.60619948,  2.95752056, -0.4937432]

	words = ['it', "'s", 'just', 'incredibly', 'dull', '.']
	scores = [0.09588719,  0.138026,    0.27497576,  0.074168,    2.23141374, -0.87973352]

	output_text = html_heatmap(words, scores)
	print(output_text)



if __name__ == '__main__':
	main()




