package ai.greatfree.modernconv;

import java.io.IOException;
import java.util.Arrays;
import java.util.Map;

import org.apache.commons.lang3.ArrayUtils;

import ai.djl.metric.Metrics;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.translate.TranslateException;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.api.LinePlot;

/**
 * Created: 08/27/2024, Bing Li
 */
final class ModernUtils
{
	public static double trainViaAlexNet(ArrayDataset trainIter, ArrayDataset testIter, int numEpochs, Trainer trainer, double[] epochCount, Map<String, double[]> evaluatorMetrics) throws IOException, TranslateException
	{
		double avgTrainTimePerEpoch = 0;
		trainer.setMetrics(new Metrics());

		EasyTrain.fit(trainer, numEpochs, trainIter, testIter);

		Metrics metrics = trainer.getMetrics();

		trainer.getEvaluators().stream().forEach(evaluator ->
		{
			evaluatorMetrics.put("train_epoch_" + evaluator.getName(), metrics.getMetric("train_epoch_" + evaluator.getName()).stream().mapToDouble(x -> x.getValue().doubleValue()).toArray());
			evaluatorMetrics.put("validate_epoch_" + evaluator.getName(), metrics.getMetric("validate_epoch_" + evaluator.getName()).stream().mapToDouble(x -> x.getValue().doubleValue()).toArray());
		});

		avgTrainTimePerEpoch = metrics.mean("epoch");

		double[] trainLoss = evaluatorMetrics.get("train_epoch_SoftmaxCrossEntropyLoss");
		double[] trainAccuracy = evaluatorMetrics.get("train_epoch_Accuracy");
		double[] testAccuracy = evaluatorMetrics.get("validate_epoch_Accuracy");

		System.out.printf("loss %.3f,", trainLoss[numEpochs - 1]);
		System.out.printf(" train acc %.3f,", trainAccuracy[numEpochs - 1]);
		System.out.printf(" test acc %.3f\n", testAccuracy[numEpochs - 1]);
		System.out.printf("%.1f examples/sec \n", trainIter.size() / (avgTrainTimePerEpoch / Math.pow(10, 9)));

		String[] lossLabel = new String[trainLoss.length + testAccuracy.length + trainAccuracy.length];

		Arrays.fill(lossLabel, 0, trainLoss.length, "train loss");
		Arrays.fill(lossLabel, trainAccuracy.length, trainLoss.length + trainAccuracy.length, "train acc");
		Arrays.fill(lossLabel, trainLoss.length + trainAccuracy.length, trainLoss.length + testAccuracy.length + trainAccuracy.length, "test acc");

		Table data = Table.create("Data").addColumns(DoubleColumn.create("epoch", ArrayUtils.addAll(epochCount, ArrayUtils.addAll(epochCount, epochCount))), DoubleColumn.create("metrics", ArrayUtils.addAll(trainLoss, ArrayUtils.addAll(trainAccuracy, testAccuracy))), StringColumn.create("lossLabel", lossLabel));

		Plot.show(LinePlot.create("", data, "epoch", "metrics", "lossLabel"));
		return avgTrainTimePerEpoch;
	}
}
