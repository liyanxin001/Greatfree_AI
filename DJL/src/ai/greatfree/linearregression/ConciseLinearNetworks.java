package ai.greatfree.linearregression;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import ai.djl.Model;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.ParameterList;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;

/**
 * Created: 08/13/2024, Bing Li
 */
final class ConciseLinearNetworks
{
	public static void main(String[] args) throws IOException, TranslateException
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray trueW = manager.create(new float[] { 2, -3.4f });
			float trueB = 4.2f;

			DataPoints dp = SyntheticData.create(manager, trueW, trueB, 1000);
			NDArray features = dp.getX();
			NDArray labels = dp.getY();

			int batchSize = 10;
			ArrayDataset dataset = LNTraining.loadArray(features, labels, batchSize, false);

			Batch batch = dataset.getData(manager).iterator().next();
			NDArray x = batch.getData().head();
			NDArray y = batch.getLabels().head();
			System.out.println(x);
			System.out.println(y);
			batch.close();

			Model model = Model.newInstance("lin-reg");

			SequentialBlock net = new SequentialBlock();
			Linear linearBlock = Linear.builder().optBias(true).setUnits(1).build();
			net.add(linearBlock);

			model.setBlock(net);

			Loss l2loss = Loss.l2Loss();

			Tracker lrt = Tracker.fixed(0.03f);
			Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

			DefaultTrainingConfig config = new DefaultTrainingConfig(l2loss).optOptimizer(sgd) // Optimizer (loss
																								// function)
					.optDevices(manager.getEngine().getDevices(1)) // single GPU
					.addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

			Trainer trainer = model.newTrainer(config);

			trainer.initialize(new Shape(batchSize, 2));

			Metrics metrics = new Metrics();
			trainer.setMetrics(metrics);

			int numEpochs = 3;
			for (int epoch = 1; epoch <= numEpochs; epoch++)
			{
				System.out.printf("Epoch %d\n", epoch);
				// Iterate over dataset
				for (Batch entry : trainer.iterateDataset(dataset))
				{
					// Update loss and evaulator
					EasyTrain.trainBatch(trainer, entry);
					// Update parameters
					trainer.step();
					entry.close();
				}
				// reset training and validation evaluators at end of epoch
				trainer.notifyListeners(listener -> listener.onEpoch(trainer));
			}

			Block layer = model.getBlock();
			ParameterList params = layer.getParameters();
			NDArray wParam = params.valueAt(0).getArray();
			NDArray bParam = params.valueAt(1).getArray();

			float[] w = trueW.sub(wParam.reshape(trueW.getShape())).toFloatArray();
			System.out.printf("Error in estimating w: [%f %f]\n", w[0], w[1]);
			System.out.printf("Error in estimating b: %f\n", trueB - bParam.getFloat());

			Path modelDir = Paths.get("/Users/libing/Temp/models/lin-reg");
			Files.createDirectories(modelDir);
			model.setProperty("Epoch", Integer.toString(numEpochs)); // save epochs trained as metadata
			model.save(modelDir, "lin-reg");
		}
	}
}
