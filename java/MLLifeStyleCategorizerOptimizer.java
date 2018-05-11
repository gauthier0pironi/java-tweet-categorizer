package ca.etsmtl.tweetprocessor.categorization.impl;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.regex.Pattern;

import net.sf.javaml.classification.AbstractClassifier;
import net.sf.javaml.core.DenseInstance;

import org.jsoup.HttpStatusException;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;

import weka.core.stemmers.SnowballStemmer;
import ca.etsmtl.tweetprocessor.categorization.Categorizer;
import ca.etsmtl.tweetprocessor.categorization.impl.MLLifeStyle.MlTools;
import ca.etsmtl.tweetprocessor.categorization.impl.MLLifeStyle.NlpTools;
import ca.etsmtl.tweetprocessor.categorization.impl.MLLifeStyle.PcaTools;
import ca.etsmtl.tweetprocessor.categorization.impl.MLLifeStyle.TweetCleaner;
import ca.etsmtl.tweetprocessor.categorization.impl.MLLifeStyle.UrlTools;

public class MLLifeStyleCategorizerOptimizer implements Categorizer, Serializable {

	private static final long serialVersionUID = 1L;

	// for Testing...
	// private static Double C_parameter = 1.0;
	// private static Double gamma_parameter = 1.0;
	// private static Double coef0_parameter = 1.0;
	// private static int degree_parameter = 1;
	// private static Double parameter1 = 1.0;
	// private static Double parameter2 = 2.0;
	// private static Double parameter3 = 2.0;
	// private static int parameter4 = 0;
	// File fileToCreateTest = new File("output/optimize.txt");
	// PrintWriter printWriterTest = null;

	// File to read with tweetClass(Integer) & tweet(String)
	private static String filePath = "lists/category_training_set.data";
	private static String fileSepartor = "\\t";
	private static int positionOfTweetInFile = 0;
	private static int positionOfTweetClassInFile = 2;

	// NLP & NGram Model
	private static boolean REMOVE_TOKEN_WITH_THRESHOLD = true;
	private static int N_GRAM_MINIMUM_THRESHOLD = 2;
	private static int N_GRAM_MAXIMUM_THRESHOLD = 1000;
	private static boolean APPLY_PCA = false;
	private static Integer PCA_NUMBER_OF_PC_TO_RETAIN = 300;
	private static boolean USE_STOP_WORDS_LIST = true;
	private static String stopWordsFilePath = "lists/stop_words.lst";
	private static boolean READ_URL = true;
	private static boolean USE_DICTIONARIES = true;
	private static String dictionary1Path = "lists/category_arts-entertainment.lst";
	private static String dictionary2Path = "lists/category_beauty-fashion.lst";
	private static String dictionary3Path = "lists/category_food.lst";
	private static boolean USE_IDF = true;
	private static boolean USE_PORTER_STEMMING = true;
	private static boolean PRINT_FEATURES_IN_FILE = false;

	// Machine Learning
	private static String naiveBayesClassifier = "NAIVES_BAYES";
	private static String svmClassifier = "SVM";
	private static String zeroRClassifier = "ZERO_R";
	private static String knnClassifier = "KNN";
	private static Integer knnClassifier_k_parameter = 3;
	private static String CLASSIFIER_TYPE = svmClassifier; // naiveBayesClassifier
	// svmClassifier
	// zeroRClassifier
	// knnClassifier
	private static String DATA_DELIMITER = ",";
	private static String classifierPerformancePath = "output/classifierPerformance.txt";
	private static int numberOfFold = 10;

	// Code things...
	private static String trainingDatasetPath = "tmp/training-features.csv";
	private static AbstractClassifier classifier;
	private static int numberOfTweets;
	private static TreeMap<String, Double> Features;
	private static TreeMap<String, Double> IDF;
	private static SnowballStemmer stemmer = new SnowballStemmer();
	private static Integer processedTweets = 0;
	DecimalFormat df_performance = new DecimalFormat("00.00");
	private static ArrayList<String> list1;
	private static ArrayList<String> list2;
	private static ArrayList<String> list3;

	public int categorize(String text) {
		TreeMap<String, Double> tweetVector = new TreeMap<String, Double>();
		Double list1count = 0.0;
		Double list2count = 0.0;
		Double list3count = 0.0;
		Double listURL1count = 0.0;
		Double listURL2count = 0.0;
		Double listURL3count = 0.0;
		Double listMetaURL1count = 0.0;
		Double listMetaURL2count = 0.0;
		Double listMetaURL3count = 0.0;
		Double listContentURL1count = 0.0;
		Double listContentURL2count = 0.0;
		Double listContentURL3count = 0.0;

		text = TweetCleaner.clean(text, READ_URL);
		String[] tokens = text.split(" ");
		// fill new Map with all features with count to 0.
		for (String key : Features.keySet()) {
			tweetVector.put(key, 0.0);
		}
		for (String token : tokens) {
			if (READ_URL && Pattern.matches("https?://t.co/[\\S]+", token)) {
				String[] words = {};
				String[] URLwords = {};
				try {
					Document doc = Jsoup.connect(token).get();

					// URL
					URLwords = UrlTools.tokenize(doc.location());

					for (String tokenURL : URLwords) {
						if (list1.contains(tokenURL)) {
							listMetaURL1count++;
						}
						if (list2.contains(tokenURL)) {
							listMetaURL2count++;
						}
						if (list3.contains(tokenURL)) {
							listMetaURL3count++;
						}
					}

					// balise <p>
					for (Element meta : doc.select("p")) {
						words = TweetCleaner.clean(meta.html(), false).split(
								" ");
						for (String word : words) {
							if (list1.contains(word)) {
								listContentURL1count++;
							}
							if (list2.contains(word)) {
								listContentURL2count++;
							}
							if (list3.contains(word)) {
								listContentURL3count++;
							}
						}
					}

					// balise <meta>
					for (Element meta : doc.select("meta")) {
						String metaName = meta.attr("name");

						// we take only meta name != ""
						if (!metaName.equals("")) {
							words = TweetCleaner.clean(meta.attr("content"),
									false).split(" ");
							for (String word : words) {
								if (list1.contains(word)) {
									listURL1count++;
								}
								if (list2.contains(word)) {
									listURL2count++;
								}
								if (list3.contains(word)) {
									listURL3count++;
								}
							}
						}
					}
				} catch (HttpStatusException e) {
					URLwords = UrlTools.tokenize(e.getUrl());
					for (String tokenURL : URLwords) {
						if (list1.contains(tokenURL)) {
							listMetaURL1count++;
						}
						if (list2.contains(tokenURL)) {
							listMetaURL2count++;
						}
						if (list3.contains(tokenURL)) {
							listMetaURL3count++;
						}
					}
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
			// to lower case once we treated the URL.
			if (READ_URL) {
				token = token.toLowerCase();
			}

			if ((USE_STOP_WORDS_LIST && !NlpTools.isStopWord(token))
					|| !USE_STOP_WORDS_LIST) {

				if (USE_DICTIONARIES && list1.contains(token)) {
					list1count++;
				}
				if (USE_DICTIONARIES && list2.contains(token)) {
					list2count++;
				}
				if (USE_DICTIONARIES && list3.contains(token)) {
					list3count++;
				}

				if (USE_PORTER_STEMMING) {
					token = stemmer.stem(token);
				}
				Double c = tweetVector.get(token);
				if (c != null && USE_IDF) {
					tweetVector.put(token, c + IDF.get(token));
				} else if (c != null) {
					tweetVector.put(token, c + 1.0);
				}
			}
		}

		Integer offset = 0;
		if (USE_DICTIONARIES) {
			offset += 3;
		}

		if (READ_URL) {
			offset += 9;
		}

		double[] values = new double[tweetVector.size() + offset];
		Set set = tweetVector.entrySet();
		Iterator i = set.iterator();
		int s = 0;
		while (i.hasNext()) {
			values[s] = (Double) ((Map.Entry) i.next()).getValue();
			s++;
		}

		// add the list/dict meta-attributes.
		if (USE_DICTIONARIES) {
			values[s++] = list1count;
			values[s++] = list2count;
			values[s++] = list3count;
		}

		if (READ_URL) {
			values[s++] = listMetaURL1count;
			values[s++] = listMetaURL2count;
			values[s++] = listMetaURL3count;
			values[s++] = listURL1count;
			values[s++] = listURL2count;
			values[s++] = listURL3count;
			values[s++] = listContentURL1count;
			values[s++] = listContentURL2count;
			values[s++] = listContentURL3count;
		}

		return Integer.parseInt((String) classifier.classify(new DenseInstance(
				values)));
	}

	public void prepare() {
		try {
			if (USE_STOP_WORDS_LIST) {
				NlpTools.computeStopWords(stopWordsFilePath);
			}
			createModelAndDumpToFile();

			// printWriterTest = new PrintWriter(new
			// FileWriter(fileToCreateTest,
			// true));

			// Testing purpose
			// for (parameter = 0.12; parameter <= 0.14; parameter+=0.005) {
			// // C_parameter = Math.pow(10, parameter);
			// C_parameter = parameter;
			// printWriterTest.print("C : " + C_parameter);
			// System.out.print("C : " + C_parameter);
			// trainClassifier();
			// crossValidation();
			// }
			//

			// for (parameter1 = -1.0; parameter1 <= 10.0; parameter1 += 0.1) {
			// for (parameter2 = -10.0; parameter2 <= 0.0; parameter2 += 1)
			// {
			// for (parameter3 = 1.0; parameter3 <= 4.0; parameter3 +=
			// 2) {
			// for (parameter4 = 2; parameter4 <= 4; parameter4++) {

			// C_parameter = Math.pow(10, parameter1);
			// gamma_parameter = Math.pow(10, parameter2);
			// C_parameter = 100.0;
			// gamma_parameter = 0.0010;
			// degree_parameter = 2;
			// coef0_parameter = parameter1;
			// System.out.print(C_parameter + "\t" + gamma_parameter
			// + "\t" + coef0_parameter + "\t"
			// + degree_parameter + "\t");
			// printWriterTest.print(C_parameter + "\t"
			// + gamma_parameter + "\t" + coef0_parameter
			// + "\t" + degree_parameter + "\t");

			// System.out.print(C_parameter + "\t" + gamma_parameter + "\t"
			// + coef0_parameter + "\t");
			// printWriterTest.print(C_parameter + "\t" + gamma_parameter
			// + "\t" + coef0_parameter + "\t");

			if (APPLY_PCA) {
				PcaTools.applyPCA(trainingDatasetPath,
						PCA_NUMBER_OF_PC_TO_RETAIN);
			}

			classifier = MlTools.trainClassifier(knnClassifier_k_parameter,
					classifier, CLASSIFIER_TYPE, trainingDatasetPath,
					APPLY_PCA, DATA_DELIMITER, PCA_NUMBER_OF_PC_TO_RETAIN);

			MlTools.crossValidation(processedTweets, CLASSIFIER_TYPE,
					classifierPerformancePath, classifier, numberOfFold);
			// }
			// }
			// }
			// printWriterTest.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private void getFeatures() throws IOException {
		File fileToCreate = new File(classifierPerformancePath);
		PrintWriter printWriter = new PrintWriter(new FileWriter(fileToCreate,
				true));
		BufferedReader buff = new BufferedReader(new FileReader(filePath));
		try {
			String line;
			while ((line = buff.readLine()) != null) {
				List<String> tokensToAddToIDF = new ArrayList<String>();
				String[] temp = line.split(fileSepartor);
				String tweet = temp[positionOfTweetInFile];
				tweet = TweetCleaner.clean(tweet, false);
				String[] tokens = tweet.split(" ");
				for (String token : tokens) {
					if ((USE_STOP_WORDS_LIST && !NlpTools.isStopWord(token))
							|| !USE_STOP_WORDS_LIST) {
						if (USE_PORTER_STEMMING) {
							token = stemmer.stem(token);
						}
						Double c = Features.get(token);
						if (c == null) {
							Features.put(token, 1.0);
						} else {
							Features.put(token, c + 1.0);
						}
						if (USE_IDF && !tokensToAddToIDF.contains(token)) {
							tokensToAddToIDF.add(token);
						}
					}
				}
				if (USE_IDF) {
					// insert the unique list of token in the IDF.
					Iterator i = tokensToAddToIDF.iterator();
					while (i.hasNext()) {
						String tokenToAdd = (String) i.next();
						Double c = IDF.get(tokenToAdd);
						if (c == null) {
							IDF.put(tokenToAdd, 1.0);
						} else {
							IDF.put(tokenToAdd, c + 1.0);
						}
					}
				}
				processedTweets++;
			}
		} finally {
			buff.close();
		}
		numberOfTweets = processedTweets;
		printWriter.println(new Date());
		printWriter.println("\n***NGram***");
		if (USE_IDF) {
			printWriter.println("IDF is used");
		} else {
			printWriter.println("IDF is not used");
		}
		if (USE_PORTER_STEMMING) {
			printWriter.println("Porter's Stemming is used");
		} else {
			printWriter.println("Porter's Stemming is not used");
		}
		if (!USE_DICTIONARIES) {
			printWriter.println("Dictionaries are not used.");
		} else {
			printWriter.println("Dictionaries are used.");
		}
		if (!READ_URL) {
			printWriter.println("URL are not used.");
		} else {
			printWriter.println("URL are used.");
		}
		if (USE_STOP_WORDS_LIST) {
			printWriter.println("Stop Words List is used");
		} else {
			printWriter.println("Stop Words List is not used");
		}
		printWriter.println("Number of tweets : " + processedTweets);
		printWriter.println("Nombre of tokens (no pruning) : "
				+ Features.size());
		printWriter.close();
	}

	private void writeFeaturesInFile() throws IOException {
		BufferedReader buff2 = new BufferedReader(new FileReader(filePath));
		File fileToCreateTrainingDataset = new File(trainingDatasetPath);
		PrintWriter printWriterDataset = new PrintWriter(
				fileToCreateTrainingDataset);

		Integer tweetComputed = 1;
		Double list1count = 0.0;
		Double list2count = 0.0;
		Double list3count = 0.0;

		Double listURL1count = 0.0;
		Double listURL2count = 0.0;
		Double listURL3count = 0.0;

		Double listMetaURL1count = 0.0;
		Double listMetaURL2count = 0.0;
		Double listMetaURL3count = 0.0;

		Double listContentURL1count = 0.0;
		Double listContentURL2count = 0.0;
		Double listContentURL3count = 0.0;
		// add header in file if we use PCA

		if (APPLY_PCA) {
			String HeaderLineToWrite = "";
			Integer i = 0;
			HeaderLineToWrite = "class";
			for (i = 1; i < Features.size() + 1; i++) {
				HeaderLineToWrite += ",a" + i;
			}
			// TODO add attributes if we use URL_READER and others things...
			if (USE_DICTIONARIES) {
				HeaderLineToWrite += ",aD1,aD2,aD3";
			}
			printWriterDataset.println(HeaderLineToWrite);
		}

		try {
			String line;
			while ((line = buff2.readLine()) != null) {
				System.out.println(tweetComputed++);
				TreeMap<String, Double> tweetVector = new TreeMap<String, Double>();
				String[] temp = line.split(fileSepartor);
				list1count = 0.0;
				list2count = 0.0;
				list3count = 0.0;
				listURL1count = 0.0;
				listURL2count = 0.0;
				listURL3count = 0.0;
				listMetaURL1count = 0.0;
				listMetaURL2count = 0.0;
				listMetaURL3count = 0.0;
				listContentURL1count = 0.0;
				listContentURL2count = 0.0;
				listContentURL3count = 0.0;

				try {
					String tweetClass = temp[positionOfTweetClassInFile];
					String tweet = temp[positionOfTweetInFile];
					tweet = TweetCleaner.clean(tweet, READ_URL);
					String[] tokens = tweet.split(" ");
					// fill new Map with all features with count to 0.
					for (String key : Features.keySet()) {
						tweetVector.put(key, 0.0);
					}
					for (String token : tokens) {

						if (READ_URL
								&& Pattern.matches("https?://t.co/[\\S]+",
										token)) {
							String[] words = {};
							String[] URLwords = {};

							try {
								Document doc = Jsoup.connect(token).get();

								// URL
								URLwords = UrlTools.tokenize(doc.location());

								for (String tokenURL : URLwords) {
									if (list1.contains(tokenURL)) {
										listMetaURL1count++;
									}
									if (list2.contains(tokenURL)) {
										listMetaURL2count++;
									}
									if (list3.contains(tokenURL)) {
										listMetaURL3count++;
									}
								}

								// balise <p>
								for (Element meta : doc.select("p")) {
									words = TweetCleaner.clean(meta.html(),
											false).split(" ");

									for (String word : words) {
										if (list1.contains(word)) {
											listContentURL1count++;
										}
										if (list2.contains(word)) {
											listContentURL2count++;
										}
										if (list3.contains(word)) {
											listContentURL3count++;
										}
									}
								}

								// balise <meta>
								for (Element meta : doc.select("meta")) {
									String metaName = meta.attr("name");

									// we take only meta name != ""
									if (!metaName.equals("")) {
										words = TweetCleaner.clean(
												meta.attr("content"), false)
												.split(" ");

										for (String word : words) {
											if (list1.contains(word)) {
												listURL1count++;
											}
											if (list2.contains(word)) {
												listURL2count++;
											}
											if (list3.contains(word)) {
												listURL3count++;
											}
										}
									}
								}
							} catch (HttpStatusException e) {
								URLwords = UrlTools.tokenize(e.getUrl());
								for (String tokenURL : URLwords) {
									if (list1.contains(tokenURL)) {
										listMetaURL1count++;
									}
									if (list2.contains(tokenURL)) {
										listMetaURL2count++;
									}
									if (list3.contains(tokenURL)) {
										listMetaURL3count++;
									}
								}
							} catch (Exception e) {
								e.printStackTrace();
							}
						}
						if (READ_URL) {
							token = token.toLowerCase();
						}

						if ((USE_STOP_WORDS_LIST && !NlpTools.isStopWord(token))
								|| !USE_STOP_WORDS_LIST) {

							if (USE_DICTIONARIES && list1.contains(token)) {
								list1count++;
							}
							if (USE_DICTIONARIES && list2.contains(token)) {
								list2count++;
							}
							if (USE_DICTIONARIES && list3.contains(token)) {
								list3count++;
							}

							if (USE_PORTER_STEMMING) {
								token = stemmer.stem(token);
							}
							Double c = tweetVector.get(token);
							if (c != null && USE_IDF) {
								tweetVector.put(token, c + IDF.get(token));
							} else if (c != null) {
								tweetVector.put(token, c + 1.0);
							}
						}

					}

					String lineToWrite = tweetClass;
					Set set = tweetVector.entrySet();
					Iterator i = set.iterator();
					while (i.hasNext()) {
						Map.Entry me = (Map.Entry) i.next();
						lineToWrite += "," + me.getValue();
					}
					if (USE_DICTIONARIES) {
						lineToWrite += "," + list1count + "," + list2count
								+ "," + list3count;
					}

					if (READ_URL) {
						lineToWrite += "," + listMetaURL1count + ","
								+ listMetaURL2count + "," + listMetaURL3count;

						lineToWrite += "," + listURL1count + ","
								+ listURL2count + "," + listURL3count;

						lineToWrite += "," + listContentURL1count + ","
								+ listContentURL2count + ","
								+ listContentURL3count;
					}
					printWriterDataset.println(lineToWrite);
				} catch (ArrayIndexOutOfBoundsException e) {
					e.printStackTrace();
				}
			}
		} finally {
			buff2.close();
			printWriterDataset.close();
		}
	}

	private void createModelAndDumpToFile() throws IOException {
		Features = new TreeMap<String, Double>();
		IDF = new TreeMap<String, Double>();

		getFeatures();
		if (REMOVE_TOKEN_WITH_THRESHOLD) {
			NlpTools.removeTokenWithThreshold(Features,
					classifierPerformancePath, N_GRAM_MINIMUM_THRESHOLD,
					N_GRAM_MAXIMUM_THRESHOLD);
		}
		if (USE_DICTIONARIES || READ_URL) {
			list1 = NlpTools.computeDictionnaryList(dictionary1Path);
			list2 = NlpTools.computeDictionnaryList(dictionary2Path);
			list3 = NlpTools.computeDictionnaryList(dictionary3Path);
		}
		if (PRINT_FEATURES_IN_FILE) {
			NlpTools.dumpFeaturesinFile(Features, classifierPerformancePath);
		}
		if (USE_IDF) {
			NlpTools.processTFIDF(IDF, numberOfTweets);
		}
		writeFeaturesInFile();
	}

}
