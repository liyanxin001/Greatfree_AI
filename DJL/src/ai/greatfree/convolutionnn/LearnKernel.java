package ai.greatfree.convolutionnn;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterList;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.NormalInitializer;
import ai.djl.training.loss.Loss;

/**
 * Created: 08/20/2024, Bing Li
 */
final class LearnKernel
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray X = manager.ones(new Shape(6, 8));
			X.set(new NDIndex(":" + "," + 2 + ":" + 6), 0f);
			System.out.println(X);

			System.out.println("------------------------");
			NDArray K = manager.create(new float[] { 1, -1 }, new Shape(1, 2));
			System.out.println(K);

			System.out.println("------------------------");
			NDArray Y = ConvolutionUtils.corr2d(manager, X, K);
			System.out.println(Y);

			System.out.println("------------------------");
			X = X.reshape(1, 1, 6, 8);
			System.out.println(X);
			System.out.println("------------------------");
			Y = Y.reshape(1, 1, 6, 7);
			System.out.println(Y);

			Loss l2Loss = Loss.l2Loss();

			// Construct a two-dimensional convolutional layer with 1 output channel and a
			// kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
			Block block = Conv2d.builder().setKernelShape(new Shape(1, 2)).optBias(false).setFilters(1).build();

			block.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
			block.initialize(manager, DataType.FLOAT32, X.getShape());

			// The two-dimensional convolutional layer uses four-dimensional input and
			// output in the format of (example, channel, height, width), where the batch
			// size (number of examples in the batch) and the number of channels are both 1

			ParameterList params = block.getParameters();
			NDArray wParam = params.get(0).getValue().getArray();
			wParam.setRequiresGradient(true);

			NDArray lossVal = null;
			ParameterStore parameterStore = new ParameterStore(manager, false);

			System.out.println("------------------------");
			for (int i = 0; i < 10; i++)
			{
				wParam.setRequiresGradient(true);
				try (GradientCollector gc = Engine.getInstance().newGradientCollector())
				{
					NDArray yHat = block.forward(parameterStore, new NDList(X), true).singletonOrThrow();
					NDArray l = l2Loss.evaluate(new NDList(Y), new NDList(yHat));
					lossVal = l;
					gc.backward(l);
				}
				// Update the kernel
				wParam.subi(wParam.getGradient().mul(0.40f));

				if ((i + 1) % 2 == 0)
				{
					System.out.println("batch " + (i + 1) + " loss: " + lossVal.sum().getFloat());
				}
			}
			System.out.println("------------------------");
			System.out.println(wParam);
		}
	}
}
