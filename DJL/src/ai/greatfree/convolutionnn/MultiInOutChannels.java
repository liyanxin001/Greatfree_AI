package ai.greatfree.convolutionnn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

/**
 * Created: 08/21/2024, Bing Li
 */
final class MultiInOutChannels
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray X = manager.create(new Shape(2, 3, 3), DataType.INT32);
			X.set(new NDIndex(0), manager.arange(9));
			X.set(new NDIndex(1), manager.arange(1, 10));
			X = X.toType(DataType.FLOAT32, true);

			NDArray K = manager.create(new Shape(2, 2, 2), DataType.INT32);
			K.set(new NDIndex(0), manager.arange(4));
			K.set(new NDIndex(1), manager.arange(1, 5));
			K = K.toType(DataType.FLOAT32, true);
			System.out.println(ConvolutionUtils.corr2dMultiIn(manager, X, K));

			System.out.println("--------------------");
			K = NDArrays.stack(new NDList(K, K.add(1), K.add(2)));
			K.getShape();
			System.out.println(K.getShape());

			System.out.println("--------------------");
			System.out.println(ConvolutionUtils.corrMultiInOut(manager, X, K));

			System.out.println("--------------------");
			X = manager.randomUniform(0f, 1.0f, new Shape(3, 3, 3));
			System.out.println(X);
			K = manager.randomUniform(0f, 1.0f, new Shape(2, 3, 1, 1));
			System.out.println(K);
			NDArray Y1 = ConvolutionUtils.corr2dMultiInOut1x1(X, K);
			System.out.println(Y1);
			NDArray Y2 = ConvolutionUtils.corrMultiInOut(manager, X, K);
			System.out.println(Y2);
			System.out.println(Math.abs(Y1.sum().getFloat() - Y2.sum().getFloat()) < 1e-6);
		}
	}
}
