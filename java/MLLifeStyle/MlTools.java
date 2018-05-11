package ca.etsmtl.tweetprocessor.categorization.impl.MLLifeStyle;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import libsvm.LibSVM;
import net.sf.javaml.classification.AbstractClassifier;
import net.sf.javaml.classification.KDtreeKNN;
import net.sf.javaml.classification.ZeroR;
import net.sf.javaml.classification.bayes.NaiveBayesClassifier;
import net.sf.javaml.classification.evaluation.CrossValidation;
import net.sf.javaml.classification.evaluation.PerformanceMeasure;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.tools.data.FileHandler;

public class MlTools {
	static DecimalFormat df_performance = new DecimalFormat("00.00");

	private static String naiveBayesClassifier = "NAIVES_BAYES";
	private static String svmClassifier = "SVM";
	private static String zeroRClassifier = "ZERO_R";
	private static String knnClassifier = "KNN";
	private static Dataset dataset;

	public static AbstractClassifier trainClassifier(Integer knnClassifier_k_parameter, AbstractClassifier classifier,String CLASSIFIER_TYPE, String trainingDatasetPath, Boolean APPLY_PCA, String DATA_DELIMITER, Integer PCA_NUMBER_OF_PC_TO_RETAIN  ) {
		try {
			File file = new File(trainingDatasetPath);
			if (!APPLY_PCA) {
				dataset = FileHandler.loadDataset(file, 0, DATA_DELIMITER);
			} else {
				dataset = FileHandler.loadDataset(file,
						PCA_NUMBER_OF_PC_TO_RETAIN, DATA_DELIMITER);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		if (CLASSIFIER_TYPE == svmClassifier) {
			classifier = new LibSVM();
			libsvm.svm_parameter svmParameters = ((LibSVM) classifier)
					.getParameters();
			// Here set SVM Parameters
			// SVM Parameters
			svmParameters.svm_type = 0; // 0 = C_SVC 1 = NU_SVC
			// svmParameters.eps = 0.00001;
			svmParameters.C = 10000;
			svmParameters.gamma = 0.0001;
			svmParameters.coef0 = 0;



			// Kernel
			svmParameters.kernel_type = 3; // 0 = linear, 1 = polynomial, 2 =
											// RBF, 3=Sigmoid
											// svmParameters.gamma =
											// gamma_parameter;
			// svmParameters.coef0 = coef0_parameter;

			// svmParameters.degree = degree_parameter;

			((LibSVM) classifier).setParameters(svmParameters);

		} else if (CLASSIFIER_TYPE == naiveBayesClassifier) {
			Boolean laplaceCorrection = true;
			Boolean logarithmicResults = false;
			Boolean sparse = false;
			classifier = new NaiveBayesClassifier(laplaceCorrection,
					logarithmicResults, sparse);
		} else if (CLASSIFIER_TYPE == knnClassifier) {
			classifier = new KDtreeKNN(knnClassifier_k_parameter);
		} else if (CLASSIFIER_TYPE == zeroRClassifier) {
			classifier = new ZeroR();
		}

		classifier.buildClassifier(dataset);
		return classifier;
	}

	
	
	
	public static void crossValidation(Integer processedTweets,
			String CLASSIFIER_TYPE, String classifierPerformancePath,
			AbstractClassifier classifier,  Integer numberOfFold)
			throws IOException {

		CrossValidation validation = new CrossValidation(classifier);
		Map results = new HashMap<Object, PerformanceMeasure>();
		results = validation.crossValidation(dataset, numberOfFold, new Random(
				1));

		printClassifierParametersInFile(classifierPerformancePath,
				CLASSIFIER_TYPE, classifier);
		printClassifierPerformanceInFile(processedTweets, results,
				classifierPerformancePath);

	}

	private static void printClassifierParametersInFile(
			String classifierPerformancePath, String CLASSIFIER_TYPE,
			AbstractClassifier classifier) throws IOException {
		File fileToCreate = new File(classifierPerformancePath);
		PrintWriter printWriter = new PrintWriter(new FileWriter(fileToCreate,
				true));
		String lineToWrite = "";

		if (CLASSIFIER_TYPE == svmClassifier) {
			lineToWrite += "\n***SVM Classifier***" + "\n";
			lineToWrite += "\tSvm type : "
					+ ((LibSVM) classifier).getParameters().svm_type + "\n";
			lineToWrite += "\tKernel type : "
					+ ((LibSVM) classifier).getParameters().kernel_type + "\n";
			lineToWrite += "\tDegree : "
					+ ((LibSVM) classifier).getParameters().degree + "\n";
			lineToWrite += "\tC : " + ((LibSVM) classifier).getParameters().C
					+ "\n";
			lineToWrite += "\tCoef0 : "
					+ ((LibSVM) classifier).getParameters().coef0 + "\n";
			lineToWrite += "\tGamma : "
					+ ((LibSVM) classifier).getParameters().gamma + "\n";
		} else if (CLASSIFIER_TYPE == naiveBayesClassifier) {
			lineToWrite += "\n***Naives Bayes Classifier***" + "\n";
		} else if (CLASSIFIER_TYPE == knnClassifier) {
			lineToWrite += "\n***KNN Classifier***" + "\n";
		} else if (CLASSIFIER_TYPE == zeroRClassifier) {
			lineToWrite += "\n***ZeroR Classifier***" + "\n";
		}

		printWriter.println(lineToWrite);
		printWriter.close();
	}

	private static void printClassifierPerformanceInFile(
			Integer processedTweets, Map results,
			String classifierPerformancePath) throws IOException {
		File fileToCreate = new File(classifierPerformancePath);
		PrintWriter printWriter = new PrintWriter(new FileWriter(fileToCreate,
				true));

		Set set = results.entrySet();
		Iterator i = set.iterator();
		printWriter.println("***Results***");
		printWriter
				.println("\tClass:\t    Precision:\t\t   Recall:\t  Repartition:\t\t\tResults:");
		Double PrecisionCumulated = 0.0;
		Double RecallCumulated = 0.0;
		Double correctlyClassifiedData = 0.0;

		while (i.hasNext()) {
			Map.Entry<Object, PerformanceMeasure> me = (Map.Entry) i.next();
			printWriter
					.println("\t\t"
							+ me.getKey().toString()
							+ "\t\t\t"
							+ df_performance.format(me.getValue()
									.getPrecision() * 100)
							+ "%\t\t\t"
							+ df_performance
									.format(me.getValue().getRecall() * 100)
							+ "%\t\t\t"
							+ df_performance.format((me.getValue().tp + me
									.getValue().fn) / processedTweets * 100)
							+ "%\t\t\t" + me.getValue());
			PrecisionCumulated += me.getValue().getPrecision()
					* (me.getValue().tp + me.getValue().fn);
			RecallCumulated += me.getValue().getRecall()
					* (me.getValue().tp + me.getValue().fn);
			correctlyClassifiedData += me.getValue().tp;
		}
		PrecisionCumulated /= processedTweets;
		RecallCumulated /= processedTweets;
		correctlyClassifiedData = correctlyClassifiedData / processedTweets;
		printWriter.println("\tTotal:\t\t\t"
				+ df_performance.format(PrecisionCumulated * 100) + "%\t\t\t"
				+ df_performance.format(RecallCumulated * 100)
				+ "%\t\t\t  100%\t\t\t" + "[" + processedTweets
				+ "]\tAccuracy = "
				+ df_performance.format(correctlyClassifiedData * 100) + "%");
		printWriter
				.println("\n******************************************************");

		// printWriterTest.println(df_performance
		// .format(correctlyClassifiedData * 100));
		System.out.println("Accuracy : "
				+ df_performance.format(correctlyClassifiedData * 100));

		printWriter.close();
	}

}
