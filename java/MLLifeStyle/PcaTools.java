package ca.etsmtl.tweetprocessor.categorization.impl.MLLifeStyle;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintWriter;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.CSVSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.PrincipalComponents;

public class PcaTools {
	

	public static void applyPCA(String trainingDatasetPath, Integer PCA_NUMBER_OF_PC_TO_RETAIN) {
		// ON réécrit sur trainingDatasetPath pour que le ML se passe comme si
		// il n'y avait pas eu de PCA.
		try {
			File fileTrainingDataset = new File(trainingDatasetPath);

			CSVLoader loader = new CSVLoader();
			loader.setSource(fileTrainingDataset);
			Instances data = loader.getDataSet();
			data.setClassIndex(0); // on signale que l'index est à 0.

			PrincipalComponents pca = new PrincipalComponents();

			pca.setMaximumAttributes(PCA_NUMBER_OF_PC_TO_RETAIN);
			pca.setInputFormat(data);
			data = Filter.useFilter(data, pca); // Seems to work

			System.out.println("variance : " + pca.getVarianceCovered()
					+ " Max num of attributes kept :  "
					+ pca.getMaximumAttributes());

			// save the new file.
			CSVSaver saver = new CSVSaver();
			saver.setInstances(data);
			saver.setFile(fileTrainingDataset);
			saver.writeBatch();

			// suppression du Header du fichier.
			// on copie tout dans un gros String sans la première ligne
			// et on réécrit au même endroit dans le fichier.
			BufferedReader buff = new BufferedReader(new FileReader(
					fileTrainingDataset));
			String bigline = "";

			// TODO peut être pas la meilleure solution ici de tout mettre dans
			// une String.
			try {
				String line;
				// skip the first line...
				line = buff.readLine();
				while ((line = buff.readLine()) != null) {
					bigline += line + "\n";
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
			PrintWriter printWriterDataset = new PrintWriter(
					fileTrainingDataset);
			printWriterDataset.println(bigline);

			printWriterDataset.close();
			buff.close();

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
