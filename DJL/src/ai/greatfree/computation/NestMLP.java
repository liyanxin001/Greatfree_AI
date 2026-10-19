package ai.greatfree.computation;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/**
 * Created: 08/19/2024, Bing Li
 */
class NestMLP extends AbstractBlock
{
	private SequentialBlock net;
	private Block dense;

//	private Block test;

	public NestMLP()
	{
		this.net = new SequentialBlock();
		this.net.add(Linear.builder().setUnits(64).build());
		this.net.add(Activation.reluBlock());
		this.net.add(Linear.builder().setUnits(32).build());
		this.net.add(Activation.reluBlock());
		super.addChildBlock("net", this.net);
		this.dense = addChildBlock("dense", Linear.builder().setUnits(16).build());
	}

	@Override
	protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params)
	{
		NDList current = inputs;
		// Fully connected layer
		current = this.net.forward(parameterStore, current, training);
		current = this.dense.forward(parameterStore, current, training);
		current = new NDList(Activation.relu(current.singletonOrThrow()));
		return current;
	}

	@Override
	public Shape[] getOutputShapes(Shape[] inputs)
	{
		Shape[] current = inputs;
		for (Block block : this.children.values())
		{
			current = block.getOutputShapes(current);
		}
		return current;
	}

	@Override
	public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes)
	{
		Shape[] shapes = inputShapes;
		for (Block child : getChildren().values())
		{
			child.initialize(manager, dataType, shapes);
			shapes = child.getOutputShapes(shapes);
		}
	}
}
