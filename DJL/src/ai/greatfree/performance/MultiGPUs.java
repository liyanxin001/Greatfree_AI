package ai.greatfree.performance;

import java.io.IOException;

import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.pooling.Pool;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.ParameterStore;
import ai.djl.training.Trainer;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.initializer.NormalInitializer;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.Table;

/**
 * Created: 08/22/2024, Bing Li
 */
final class MultiGPUs
{
	public static void main(String[] args) throws IOException, TranslateException
	{
		int numClass = 10;
		// This model uses a smaller convolution kernel, stride, and padding and
		// removes the maximum pooling layer
		SequentialBlock net = new SequentialBlock();
		net.add(Conv2d.builder().setFilters(64).setKernelShape(new Shape(3, 3)).optPadding(new Shape(1, 1)).build()).add(BatchNorm.builder().build()).add(Activation::relu).add(PerUtils.resnetBlock(64, 2, true)).add(PerUtils.resnetBlock(128, 2, false)).add(PerUtils.resnetBlock(256, 2, false)).add(PerUtils.resnetBlock(512, 2, false)).add(Pool.globalAvgPool2dBlock()).add(Linear.builder().setUnits(numClass).build());

		System.out.println(net);

		Model model = Model.newInstance("training-multiple-gpus-1");
		model.setBlock(net);

		Loss loss = Loss.softmaxCrossEntropyLoss();

		Tracker lrt = Tracker.fixed(0.1f);
		Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

		DefaultTrainingConfig config = new DefaultTrainingConfig(loss).optOptimizer(sgd) // Optimizer (loss function)
				.optInitializer(new NormalInitializer(0.01f), Parameter.Type.WEIGHT) // setting the initializer
				.optDevices(Engine.getInstance().getDevices(1)) // setting the number of GPUs needed
				.addEvaluator(new Accuracy()) // Model Accuracy
				.addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

		Trainer trainer = model.newTrainer(config);

		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray X = manager.randomUniform(0f, 1.0f, new Shape(4, 1, 28, 28));
			trainer.initialize(X.getShape());

			NDList[] res = Batchifier.STACK.split(new NDList(X), 4, true);

			ParameterStore parameterStore = new ParameterStore(manager, true);

			System.out.println(net.forward(parameterStore, new NDList(res[0]), false).singletonOrThrow());
			System.out.println(net.forward(parameterStore, new NDList(res[1]), false).singletonOrThrow());
			System.out.println(net.forward(parameterStore, new NDList(res[2]), false).singletonOrThrow());
			System.out.println(net.forward(parameterStore, new NDList(res[3]), false).singletonOrThrow());
		}

		System.out.println("-------------------------");
		System.out.println(net.getChildren().values().get(0).getParameters().get("weight").getArray().get(new NDIndex("0:1")));

		int numEpochs = Integer.getInteger("MAX_EPOCH", 10);

		double[] testAccuracy;
		double[] epochCount;

		epochCount = new double[numEpochs];

		for (int i = 0; i < epochCount.length; i++)
		{
			epochCount[i] = (i + 1);
		}

		System.out.println("**********************");
		System.out.println("1) GPU count = " + Engine.getInstance().getGpuCount());
		System.out.println("**********************");

//		Map<String, double[]> evaluatorMetrics = new HashMap<>();
//		double avgTrainTimePerEpoch = 0;

		Table data = null;
		// We will check if we have at least 1 GPU available. If yes, we run the
		// training on 1 GPU.
		if (Engine.getInstance().getGpuCount() >= 1)
		{
			testAccuracy = PerUtils.train(numEpochs, trainer, 256);

			data = Table.create("Data");
			data = data.addColumns(DoubleColumn.create("X", epochCount), DoubleColumn.create("testAccuracy", testAccuracy));
		}

		/*
		 * The below code is designed for CPU training. 08/23/2024, Bing Li
		 */
		testAccuracy = PerUtils.train(numEpochs, trainer, 256);

		System.out.println("**********************");
		System.out.println("2) GPU count = " + Engine.getInstance().getGpuCount());
		System.out.println("**********************");
		System.out.println("testAccuracy = " + testAccuracy);

		data = Table.create("Data");
		data = data.addColumns(DoubleColumn.create("X", epochCount), DoubleColumn.create("testAccuracy", testAccuracy));
	}
}

