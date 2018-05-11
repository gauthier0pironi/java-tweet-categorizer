package ca.etsmtl.tweetprocessor.categorization.impl.MLLifeStyle;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class NlpTools {
	
	private static HashSet<String> StopWords;

	public static void computeStopWords(String stopWordsFilePath) throws IOException {
		StopWords = new HashSet<String>();
		BufferedReader buff = new BufferedReader(new FileReader(
				stopWordsFilePath));
		String line;
		try {
			while ((line = buff.readLine()) != null) {
				StopWords.add(line);
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			buff.close();
		}
	}
	
	public static boolean isStopWord(String token) {
		return StopWords.contains(token);
	}

	
	public static ArrayList<String> computeDictionnaryList(String dictionaryPath)
			throws IOException {

		ArrayList<String> list = new ArrayList<String>();

		BufferedReader buff = new BufferedReader(new FileReader(dictionaryPath));
		try {
			String token;
			while ((token = buff.readLine()) != null) {
				try {
					list.add(token);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		} finally {
			buff.close();
		}

		return list;
	}

	public static void removeTokenWithThreshold(
			TreeMap<String, Double> Features, String classifierPerformancePath,
			Integer N_GRAM_MINIMUM_THRESHOLD, Integer N_GRAM_MAXIMUM_THRESHOLD)
			throws IOException {
		File fileToCreate = new File(classifierPerformancePath);
		PrintWriter printWriter = new PrintWriter(new FileWriter(fileToCreate,
				true));
		// eliminate less frequent than min and more frequent than max
		Iterator<Map.Entry<String, Double>> it = Features.entrySet().iterator();
		while (it.hasNext()) {
			Map.Entry<String, Double> entry = it.next();
			if (entry.getValue() < N_GRAM_MINIMUM_THRESHOLD
					|| entry.getValue() > N_GRAM_MAXIMUM_THRESHOLD) {
				it.remove();
			}
		}
		printWriter.println("Pruning : Only tokens appearing between "
				+ N_GRAM_MINIMUM_THRESHOLD + " and " + N_GRAM_MAXIMUM_THRESHOLD
				+ " times are kept.");
		printWriter.println("Number of tokens after pruning : "
				+ Features.size());
		printWriter.close();
	}

	public static void processTFIDF(TreeMap<String, Double> IDF,
			Integer numberOfTweets) {
		// compute IDF with idf = log(nb_total/nb_doc)
		Iterator<Map.Entry<String, Double>> itIDF = IDF.entrySet().iterator();
		while (itIDF.hasNext()) {
			Map.Entry<String, Double> entry = itIDF.next();
			IDF.put(entry.getKey(),
					Math.log10(numberOfTweets / entry.getValue()));
		}

	}
	

	public static void dumpFeaturesinFile(TreeMap<String, Double> Features,String classifierPerformancePath) throws IOException {
		File performanceFileToWrite = new File(classifierPerformancePath);
		PrintWriter printWriter = new PrintWriter(new FileWriter(
				performanceFileToWrite, true));
		// to have list sorted by values not by keys
		List list = new LinkedList(Features.entrySet());
		Collections.sort(list, new Comparator() {
			public int compare(Object o2, Object o1) {
				return ((Comparable) ((Map.Entry) (o1)).getValue())
						.compareTo(((Map.Entry) (o2)).getValue());
			}
		});
		Iterator i = list.iterator();
		while (i.hasNext()) {
			Map.Entry me = (Map.Entry) i.next();
			Double value = (Double) me.getValue();
			if (value != 0) {
				printWriter.print(me.getKey() + ":");
				printWriter.print(value.intValue() + "|");
			}
		}
		printWriter.println("");
		printWriter.close();
	}


}
