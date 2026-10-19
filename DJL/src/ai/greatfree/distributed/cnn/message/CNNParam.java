package ai.greatfree.distributed.cnn.message;

import java.io.Serializable;

public class CNNParam implements Serializable
{
	private static final long serialVersionUID = 6554832986908324197L;
	
	private byte[] params;
	
	public CNNParam(byte[] params)
	{
		this.params = params;
	}

	public byte[] getParams()
	{
		return this.params;
	}
}
