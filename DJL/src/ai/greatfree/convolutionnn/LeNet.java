package ai.greatfree.convolutionnn;

import java.io.IOException;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.pooling.Pool;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;

/**
 * Created: 08/21/2024, Bing Li
 */
final class LeNet
{
	public static void main(String[] args) throws IOException, TranslateException
	{
		Engine.getInstance().setRandomSeed(1111);
		SequentialBlock block = new SequentialBlock();
		block.add(Conv2d.builder().setKernelShape(new Shape(5, 5)).optPadding(new Shape(2, 2)).optBias(false).setFilters(6).build()).add(Activation::sigmoid).add(Pool.avgPool2dBlock(new Shape(5, 5), new Shape(2, 2), new Shape(2, 2))).add(Conv2d.builder().setKernelShape(new Shape(5, 5)).setFilters(16).build()).add(Activation::sigmoid).add(Pool.avgPool2dBlock(new Shape(5, 5), new Shape(2, 2), new Shape(2, 2)))
				// Blocks.batchFlattenBlock() will transform the input of the shape (batch size,
				// channel, height, width)
				// into the input of the shape (batch size, channel * height * width)
				.add(Blocks.batchFlattenBlock()).add(Linear.builder().setUnits(120).build()).add(Activation::sigmoid).add(Linear.builder().setUnits(84).build()).add(Activation::sigmoid).add(Linear.builder().setUnits(10).build());
		System.out.println(block);

		float lr = 0.9f;
		Model model = Model.newInstance("cnn");
		model.setBlock(block);

		Loss loss = Loss.softmaxCrossEntropyLoss();

		Tracker lrt = Tracker.fixed(lr);
		Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

		DefaultTrainingConfig config = new DefaultTrainingConfig(loss).optOptimizer(sgd) // Optimizer (loss function)
				.optDevices(Engine.getInstance().getDevices(1)) // Single GPU
				.addEvaluator(new Accuracy()) // Model Accuracy
				.addTrainingListeners(TrainingListener.Defaults.basic());

		Trainer trainer = model.newTrainer(config);
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray X = manager.randomUniform(0f, 1.0f, new Shape(1, 1, 28, 28));
			trainer.initialize(X.getShape());

			Shape currentShape = X.getShape();

			for (int i = 0; i < block.getChildren().size(); i++)
			{
				Shape[] newShape = block.getChildren().get(i).getValue().getOutputShapes(new Shape[] { currentShape });
				currentShape = newShape[0];
				System.out.println(block.getChildren().get(i).getKey() + " layer output : " + currentShape);
			}
		}

		int batchSize = 256;
		int numEpochs = Integer.getInteger("MAX_EPOCH", 10);
		double[] epochCount;
		epochCount = new double[numEpochs];

		for (int i = 0; i < epochCount.length; i++)
		{
			epochCount[i] = (i + 1);
		}

		FashionMnist trainIter = FashionMnist.builder().optUsage(Dataset.Usage.TRAIN).setSampling(batchSize, true).optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE)).build();

		FashionMnist testIter = FashionMnist.builder().optUsage(Dataset.Usage.TEST).setSampling(batchSize, true).optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE)).build();

		trainIter.prepare();
		testIter.prepare();
		
		ConvolutionUtils.trainViaLeNet(trainIter, testIter, numEpochs, trainer, epochCount);
	}
}
