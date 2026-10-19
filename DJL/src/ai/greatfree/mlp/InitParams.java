package ai.greatfree.mlp;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

/**
 * Created: 08/18/2024, Bing Li
 */
class InitParams
{
	private NDArray w;
	private NDArray b;
	private NDList l;

	public InitParams(NDManager manager, int numInputs)
	{
		this.w = manager.randomNormal(0, 1.0f, new Shape(numInputs, 1), DataType.FLOAT32);
		this.b = manager.zeros(new Shape(1));
		this.w.setRequiresGradient(true);
		this.b.setRequiresGradient(true);
	}

	public NDArray getW()
	{
		return this.w;
	}

	public NDArray getB()
	{
		return this.b;
	}
	
	public NDList getList()
	{
		return this.l;
	}
}
