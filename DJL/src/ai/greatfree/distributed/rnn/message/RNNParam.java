package ai.greatfree.distributed.rnn.message;

import java.io.Serializable;

/**
 * Created: 08/29/2024, Bing Li
 */
public class RNNParam implements Serializable
{
	private static final long serialVersionUID = 6554832986908324197L;
	
	private byte[] params;
	
	public RNNParam(byte[] params)
	{
		this.params = params;
	}

	public byte[] getParams()
	{
		return this.params;
	}
}

