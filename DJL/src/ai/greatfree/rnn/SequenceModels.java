package ai.greatfree.rnn;

import java.io.IOException;

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.SequentialBlock;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.greatfree.utils.Functions;
import tech.tablesaw.plotly.Plot;

/**
 * Created: 08/27/2024, Bing Li
 */
final class SequenceModels
{
	public static void main(String[] args) throws IOException, TranslateException
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			int T = 1000; // Generate a total of 1000 points
			NDArray time = manager.arange(1f, T + 1);
			NDArray x = time.mul(0.01).sin().add(manager.randomNormal(0f, 0.2f, new Shape(T), DataType.FLOAT32));

			double[] xAxis = Functions.floatToDoubleArray(time.toFloatArray());
			double[] yAxis = Functions.floatToDoubleArray(x.toFloatArray());

			Plot.show(RNNUtils.plot(xAxis, yAxis, "time", "x"));

			int tau = 4;
			NDArray features = manager.zeros(new Shape(T - tau, tau));

			for (int i = 0; i < tau; i++)
			{
				features.set(new NDIndex(":, {}", i), x.get(new NDIndex("{}:{}", i, T - tau + i)));
			}
			NDArray labels = x.get(new NDIndex("" + tau + ":")).reshape(new Shape(-1, 1));

			int batchSize = 16;
			int nTrain = 600;
			// Only the first `nTrain` examples are used for training
			ArrayDataset trainIter = new ArrayDataset.Builder().setData(features.get(new NDIndex(":{}", nTrain))).optLabels(labels.get(new NDIndex(":{}", nTrain))).setSampling(batchSize, true).build();

			SequentialBlock net = RNNUtils.getNet();
			Model model = RNNUtils.train(net, trainIter, batchSize, 5, 0.01f);

			/**
			 * One step prediction. 08/28/2024, Bing Li
			 */
			Translator<NDList, ?> translator = new NoopTranslator();
			Predictor<NDList, ?> predictor = model.newPredictor(translator);
			NDArray oneStepPreds = ((NDList) predictor.predict(new NDList(features))).get(0);
			/*
			 * The line must be commented when running the method, multiStepPredict, of
			 * RNNUtils. 08/28/2024, Bing Li
			 */
			// Plot.show(RNNUtils.oneStepPredict(oneStepPreds, time, x, tau));

			/**
			 * Multiple steps prediction. 08/28/2024, Bing Li
			 */
			NDArray multiStepPreds = manager.zeros(new Shape(T));
			multiStepPreds.set(new NDIndex(":{}", nTrain + tau), x.get(new NDIndex(":{}", nTrain + tau)));
			for (int i = nTrain + tau; i < T; i++)
			{
				NDArray tempX = multiStepPreds.get(new NDIndex("{}:{}", i - tau, i)).reshape(new Shape(1, -1));
				NDArray prediction = ((NDList) predictor.predict(new NDList(tempX))).get(0);
				multiStepPreds.set(new NDIndex(i), prediction);
			}
			Plot.show(RNNUtils.multiStepPredict(multiStepPreds, oneStepPreds, nTrain, time, x, tau));

			/**
			 * Multiple steps prediction by computing predictions on the entire sequence. 08/28/2024, Bing Li
			 */
			int maxSteps = 64;
			features = manager.zeros(new Shape(T - tau - maxSteps + 1, tau + maxSteps));
			// Column `i` (`i` < `tau`) are observations from `x` for time steps from
			// `i + 1` to `i + T - tau - maxSteps + 1`
			for (int i = 0; i < tau; i++)
			{
				features.set(new NDIndex(":, {}", i), x.get(new NDIndex("{}:{}", i, i + T - tau - maxSteps + 1)));
			}
			// Column `i` (`i` >= `tau`) are the (`i - tau + 1`)-step-ahead predictions for
			// time steps from `i + 1` to `i + T - tau - maxSteps + 1`
			for (int i = tau; i < tau + maxSteps; i++)
			{
				NDArray tempX = features.get(new NDIndex(":, {}:{}", i - tau, i));
				NDArray prediction = ((NDList) predictor.predict(new NDList(tempX))).get(0);
				features.set(new NDIndex(":, {}", i), prediction.reshape(-1));
			}
			Plot.show(RNNUtils.multiStepPredict(time, features, maxSteps, tau, T));
		}
	}
}
