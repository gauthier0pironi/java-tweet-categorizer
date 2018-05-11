package ca.etsmtl.tweetprocessor.categorization.impl.MLLifeStyle;

public class TweetCleaner {

	public static String clean(String tweetToClean, Boolean READ_URL) {

		String tmpTweet = tweetToClean;

		// remove @username
		tmpTweet = tmpTweet.replaceAll("@\\S*", " username ");

		// replace #hashtag with hashtag
		tmpTweet = tmpTweet.replaceAll("#", "");

		if (READ_URL) {
			// remove ponctuation .,;:!?"*
			tmpTweet = tmpTweet.replaceAll("[,;()\"*]", " ");
			tmpTweet = tmpTweet.replaceAll(" . ", " ");
			tmpTweet = tmpTweet.replaceAll(" : ", " ");
		} else {
			// replace http://myurl.com with url
			tmpTweet = tmpTweet.replaceAll("https?://[\\S]*", " url ");
			// remove ponctuation .,;:!?"*
			tmpTweet = tmpTweet.replaceAll("[.:,;()\"*]", " ");
		}

		// separate "?"
		tmpTweet = tmpTweet.replaceAll("[?]", " ? ");
		tmpTweet = tmpTweet.replaceAll("[!]", " ! ");

		// remove RT
		tmpTweet = tmpTweet.replaceAll("RT ", "");

		// remove numbers
		tmpTweet = tmpTweet.replaceAll(" [\\d]{1,} ", " ");

		// replace 2+ spaces to 1 space
		tmpTweet = tmpTweet.replaceAll(" {2,}", " ");

		// we don't lower case here to avoid destroying the URL
		if (!READ_URL) {
			tmpTweet = tmpTweet.toLowerCase();
		}

		// if the first caracter is " " then remove it.
		if (!tmpTweet.equals("") && tmpTweet.substring(0, 1).equals(" ")) {
			tmpTweet = tmpTweet.substring(1);
		}

		return tmpTweet;
	}

}
