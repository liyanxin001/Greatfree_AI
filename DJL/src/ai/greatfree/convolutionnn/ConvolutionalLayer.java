package ai.greatfree.convolutionnn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

/**
 * Created: 08/20/2024, Bing Li
 */
class ConvolutionalLayer
{
	private NDArray w;
	private NDArray b;

	public ConvolutionalLayer(NDManager manager, Shape shape)
	{
		this.w = manager.create(shape);
		this.b = manager.randomNormal(new Shape(1));
		this.w.setRequiresGradient(true);
	}

	public NDArray forward(NDManager manager, NDArray X)
	{
		return ConvolutionUtils.corr2d(manager, X, this.w).add(this.b);
	}

	public NDArray getW()
	{
		return this.w;
	}

	public NDArray getB()
	{
		return this.b;
	}
}

