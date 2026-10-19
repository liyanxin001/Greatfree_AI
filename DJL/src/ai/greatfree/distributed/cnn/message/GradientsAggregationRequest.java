package ai.greatfree.distributed.cnn.message;

import java.util.List;



import edu.greatfree.container.message.Request;

/**
 * Created: 08/29/2024, Bing Li
 */
public class GradientsAggregationRequest extends Request
{
	private static final long serialVersionUID = 2838739878659316287L;
	
	private String childKey;
	private List<CNNParam> params;

	public GradientsAggregationRequest(String childKey, List<CNNParam> params)
	{
		super(CNNAppID.GRADIENTS_AGGREGATION_REQUEST);
		this.childKey = childKey;
		this.params = params;
	}
	
	public String getChildKey()
	{
		return this.childKey;
	}

	public List<CNNParam> getParams()
	{
		return this.params;
	}
}
