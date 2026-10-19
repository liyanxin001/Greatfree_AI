package ai.greatfree.convolutionnn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.pooling.Pool;
import ai.djl.training.ParameterStore;

/**
 * Created: 08/21/2024, Bing Li
 */
final class PaddingStridePooling
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray X = manager.arange(16f).reshape(1, 1, 4, 4);
			System.out.println(X);

			// defining block specifying kernel and stride
			Block block = Pool.maxPool2dBlock(new Shape(3, 3), new Shape(3, 3));
			block.initialize(manager, DataType.FLOAT32, new Shape(1, 1, 4, 4));

			ParameterStore parameterStore = new ParameterStore(manager, false);
			// Because there are no model parameters in the pooling layer, we do not need
			// to call the parameter initialization function
			System.out.println("--------------------");
			System.out.println(block.forward(parameterStore, new NDList(X), true).singletonOrThrow());

			System.out.println("--------------------");
			// redefining block shapes for kernel shape, stride shape and pad shape
			block = Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2), new Shape(1, 1));
			// block forward method
			System.out.println(block.forward(parameterStore, new NDList(X), true).singletonOrThrow());

			System.out.println("--------------------");
			// redefining block shapes for kernel shape, stride shape and pad shape
			block = Pool.maxPool2dBlock(new Shape(2, 3), new Shape(2, 3), new Shape(1, 2));
			System.out.println(block.forward(parameterStore, new NDList(X), true).singletonOrThrow());

			System.out.println("--------------------");
			X = X.concat(X.add(1), 1);
			System.out.println(X);
			
			System.out.println("--------------------");
			block = Pool.maxPool2dBlock(new Shape(3,3), new Shape(2,2), new Shape(1,1));
			System.out.println(block.forward(parameterStore, new NDList(X), true).singletonOrThrow());
		}
	}
}

