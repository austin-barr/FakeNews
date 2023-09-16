import tweepy
import json
import numpy

# Twitter Developer API tokens
auth = tweepy.OAuthHandler('IesOBzYK3dutD16fJVSbCvKf4', 'a0xDyPxZA4r6yt2lREtxTTjBmSphfQupZSpwGLRTwcEBrAQchK')
auth.set_access_token('1341511431980773387-TbS1cCKg8Puxg7BStCR1BnmF8OfR4K', '7LrZL8WltHse7OJquHkUFtFfBXmbuj3a5sGFEayQfyhUD')

api = tweepy.API(auth, wait_on_rate_limit=True)
#, wait_on_rate_limit_notify=True



#def get_tweets(lower, upper):
lower, upper = 0, 10
m, n = 0, 0
mapping = numpy.load("../../data/pol_id_twitter_mapping.pkl", allow_pickle=True)
mapping = [mapping[i] for i in range(lower,upper)]
for i, user in enumerate(mapping):  # user id to twitter id mappings {user_id: twitter_account_id}
	try:
		# get recent 200 tweets of the user
		statuses = api.user_timeline(user_id=user, count=200)
		json_object = [json.dumps(s._json) + '\n' for s in statuses]
		# write the recent 200 tweet objects into a json file
		with open(str(i+lower) + ".json", "w") as outfile:
			outfile.writelines(json_object)
		outfile.close()
	except tweepy.errors.TweepyException as err: # handle deleted/suspended accounts
		if str(err) == 'Not authorized.':  
			m+=1
			print(f'Not authorized: {m}')
		else:
			n+=1
			print(f'Page does not exist: {n}')
	print(f'user number: {i}')
    
