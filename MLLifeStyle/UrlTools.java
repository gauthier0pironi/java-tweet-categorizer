package ca.etsmtl.tweetprocessor.categorization.impl.MLLifeStyle;

public class UrlTools {

	public static String[] tokenize(String URL) {
		String URL_tmp = URL;
		String[] tokens = {};
		
		URL_tmp = cleanURL(URL_tmp);
		
		tokens= URL_tmp.split(" ");
		return tokens;
	}


	private static String cleanURL(String URL) {
		String URL_tmp = URL;
		
		URL_tmp = URL_tmp.replaceAll("http", " ");
		URL_tmp = URL_tmp.replaceAll("https", " ");
		URL_tmp = URL_tmp.replaceAll("-", " ");
		URL_tmp = URL_tmp.replaceAll(":", " ");
		URL_tmp = URL_tmp.replaceAll(";", " ");
		URL_tmp = URL_tmp.replaceAll("/", " ");
		URL_tmp = URL_tmp.replaceAll("\\.", " ");
		URL_tmp = URL_tmp.replaceAll("_", " ");
		URL_tmp = URL_tmp.replaceAll(" {2,}", " ");
		URL_tmp = URL_tmp.toLowerCase();
		if (URL_tmp.substring(0, 1).equals(" "))
		{
			URL_tmp = URL_tmp.substring(1);
		}
		
		return URL_tmp;
	}
}
