package ai.greatfree.distributed.rnn.message;

import java.util.List;

import edu.greatfree.container.message.Request;

/**
 * Created: 08/29/2024, Bing Li
 */
public class GradientsAggregationRequest extends Request
{
	private static final long serialVersionUID = 2838739878659316287L;
	
	private String childKey;
	private List<RNNParam> params;

	public GradientsAggregationRequest(String childKey, List<RNNParam> params)
	{
		super(RNNAppID.GRADIENTS_AGGREGATION_REQUEST);
		this.childKey = childKey;
		this.params = params;
	}
	
	public String getChildKey()
	{
		return this.childKey;
	}

	public List<RNNParam> getParams()
	{
		return this.params;
	}
}
