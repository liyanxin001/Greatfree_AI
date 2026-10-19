package ai.greatfree.convolutionnn;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.loss.Loss;

/**
 * Created: 08/21/2024, Bing Li
 */
final class Stride
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray X = manager.randomUniform(0f, 1.0f, new Shape(1, 1, 8, 8));

			Block block = Conv2d.builder().setKernelShape(new Shape(3, 3)).optPadding(new Shape(1, 1)).optStride(new Shape(2, 2)).setFilters(1).build();

			TrainingConfig config = new DefaultTrainingConfig(Loss.l2Loss());
			Model model = Model.newInstance("conv2D");
			model.setBlock(block);

			Trainer trainer = model.newTrainer(config);
			trainer.initialize(X.getShape());

			NDArray yHat = trainer.forward(new NDList(X)).singletonOrThrow();
			System.out.println(yHat.getShape().slice(2));

			System.out.println("--------------------");

			block = Conv2d.builder().setKernelShape(new Shape(3, 5)).optPadding(new Shape(0, 1)).optStride(new Shape(3, 4)).setFilters(1).build();

			model.setBlock(block);

			trainer = model.newTrainer(config);
			trainer.initialize(X.getShape());

			yHat = trainer.forward(new NDList(X)).singletonOrThrow();
			System.out.println(yHat.getShape().slice(2));
		}
	}
}

