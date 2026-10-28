package ai.greatfree.distributed.cnn.registry;

import java.net.SocketException;
/*     */ import java.util.HashMap;
/*     */ import java.util.Map;
/*     */ import java.util.Set;
/*     */ import org.greatfree.chat.ChatConfig;
/*     */ import org.greatfree.cluster.message.IsRootOnlineResponse;
/*     */ import org.greatfree.exceptions.PeerKeyNotExistedInRegistryException;
/*     */ import org.greatfree.framework.container.p2p.message.AllPeerNamesResponse;
/*     */ import org.greatfree.framework.container.p2p.message.AllRegisteredIPsResponse;
/*     */ import org.greatfree.framework.container.p2p.message.ChatPartnerRequest;
/*     */ import org.greatfree.framework.container.p2p.message.ChatRegistryRequest;
/*     */ import org.greatfree.framework.container.p2p.message.ClusterIPRequest;
/*     */ import org.greatfree.framework.container.p2p.message.IsRootOnlineRequest;
/*     */ import org.greatfree.framework.container.p2p.message.LeaveClusterNotification;
/*     */ import org.greatfree.framework.container.p2p.message.MulticastChildrenRequest;
/*     */ import org.greatfree.framework.container.p2p.message.MulticastChildrenResponse;
/*     */ import org.greatfree.framework.container.p2p.message.PartnersRequest;
/*     */ import org.greatfree.framework.container.p2p.message.PartnersResponse;
/*     */ import org.greatfree.framework.container.p2p.message.PeerAddressRequest;
/*     */ import org.greatfree.framework.container.p2p.message.PeerDisableStateRequest;
/*     */ import org.greatfree.framework.container.p2p.message.PeerDisableStateResponse;
/*     */ import org.greatfree.framework.container.p2p.message.PortRequest;
/*     */ import org.greatfree.framework.container.p2p.message.RegisterPeerRequest;
/*     */ import org.greatfree.framework.container.p2p.message.SearchBrokerRequest;
/*     */ import org.greatfree.framework.container.p2p.message.SearchBrokerResponse;
/*     */ import org.greatfree.framework.container.p2p.message.UnregisterPeerRequest;
/*     */ import org.greatfree.framework.p2p.RegistryConfig;
/*     */ import org.greatfree.framework.p2p.message.ChatPartnerResponse;
/*     */ import org.greatfree.framework.p2p.message.ChatRegistryResponse;
/*     */ import org.greatfree.framework.p2p.registry.AccountRegistry;
/*     */ import org.greatfree.framework.p2p.registry.PeerChatAccount;
/*     */ import org.greatfree.framework.p2p.registry.PeerRegistry;
/*     */ import org.greatfree.message.PeerAddressResponse;
/*     */ import org.greatfree.message.PortResponse;
/*     */ import org.greatfree.message.RegisterPeerResponse;
/*     */ import org.greatfree.message.UnregisterPeerResponse;
/*     */ import org.greatfree.message.multicast.ClusterIPResponse;
/*     */ import org.greatfree.server.PeerAccount;
/*     */ import org.greatfree.util.IPAddress;
/*     */ import org.greatfree.util.IPPort;
/*     */ import org.greatfree.util.Tools;
/*     */ import org.greatfree.util.UtilConfig;
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ class Register
/*     */ {
/*     */   public static void register() throws SocketException {
/*  55 */     String localIP = Tools.getLocalIP();
/*     */ 
/*     */     
/*  58 */     PeerRegistry.SYSTEM().register(ChatConfig.CHAT_REGISTRY_SERVER_KEY, ChatConfig.CHAT_REGISTRY_NAME, localIP, 8941, false, false);
/*     */     
/*  60 */     PeerRegistry.SYSTEM().registeOthers(ChatConfig.CHAT_REGISTRY_SERVER_KEY, ChatConfig.CHAT_REGISTRY_PORT_KEY, localIP, 8942);
/*     */     
/*  62 */     PeerRegistry.SYSTEM().registeOthers(ChatConfig.CHAT_REGISTRY_SERVER_KEY, RegistryConfig.REGISTRY_ADMIN_PORT_KEY, localIP, 8943);
/*     */   }
/*     */ 
/*     */ 
/*     */   
/*     */   public static void unregister() {
/*  68 */     PeerRegistry.SYSTEM().dispose();
/*     */ 
/*  70 */     AccountRegistry.APP().dispose();
/*     */   }
/*     */ 
/*     */   
/*     */   public static ChatRegistryResponse registerChat(ChatRegistryRequest req) {
/*  75 */     AccountRegistry.APP().add(new PeerChatAccount(req.getPeerID(), req.getPeerName(), req.getPeerDescription(), req.getPreference()));
/*  76 */     return new ChatRegistryResponse(true);
/*     */   }
/*     */ 
/*     */   
/*     */   public static ChatPartnerResponse checkChatPartner(ChatPartnerRequest req) {
/*     */     ChatPartnerResponse response;
/*  82 */     if (PeerRegistry.SYSTEM().isExisted(req.getPartnerKey())) {
/*     */       
/*  84 */       PeerAccount account = PeerRegistry.SYSTEM().get(req.getPartnerKey());
/*  85 */       if (AccountRegistry.APP().isAccountExisted(req.getPartnerKey()))
/*     */       {
/*  87 */         PeerChatAccount chatAccount = AccountRegistry.APP().getAccount(req.getPartnerKey());
/*  88 */         response = new ChatPartnerResponse(account.getPeerKey(), account.getPeerName(), chatAccount.getDescription(), chatAccount.getPreference(), account.getIP(), account.getPeerPort());
/*     */       }
/*     */       else
/*     */       {
/*  92 */         response = new ChatPartnerResponse(account.getPeerKey(), account.getPeerName(), "", "", account.getIP(), account.getPeerPort());
/*     */       }
/*     */     
/*     */     } else {
/*     */       
/*  97 */       response = new ChatPartnerResponse("", "", "", "", "", 0);
/*     */     } 
/*  99 */     return response;
/*     */   }
/*     */ 
/*     */ 
/*     */   
/*     */   public static PartnersResponse checkPartners(PartnersRequest req) {
/* 105 */     Map<String, IPPort> ips = new HashMap<String, IPPort>();
/* 106 */     for (String entry : req.getPartnerKeys()) {
/*     */       
/* 108 */       PeerAccount account = PeerRegistry.SYSTEM().get(entry);
/* 109 */       ips.put(entry, new IPPort(account.getIP(), account.getPeerPort()));
/*     */     } 
/* 111 */     return new PartnersResponse(ips);
/*     */   }
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */   
/*     */   public static RegisterPeerResponse registerPeer(RegisterPeerRequest req) {
/* 121 */     if (AccountRegistry.APP().isAccountExisted(req.getPeerKey()))
/*     */     {
/*     */       
/* 124 */       return new RegisterPeerResponse(true);
/*     */     }
/*     */     
/* 127 */     AccountRegistry.APP().add(new PeerChatAccount(req.getPeerKey(), req.getPeerName()));
/*     */     
/* 129 */     return new RegisterPeerResponse(PeerRegistry.SYSTEM().register(req.getPeerKey(), req.getPeerName(), req.getIP(), req.getPort(), req.isServerDisabled(), req.isBroker()));
/*     */   }
/*     */ 
/*     */ 
/*     */   
/* 134 */   public static PortResponse retrievePort(PortRequest req) { return new PortResponse(PeerRegistry.SYSTEM().getIdlePort(req.getPeerKey(), req.getPortKey(), req.getIP(), req.getCurrentPort())); }
/*     */ 
/*     */ 
/*     */ 
/*     */   
/*     */   public static ClusterIPResponse retrieveClusterIP(ClusterIPRequest req) {
/* 140 */     Set<String> childrenKeys = Clusters.SYSTEM().getChildKeys(req.getRootKey());
/* 141 */     if (childrenKeys != UtilConfig.NO_KEYS)
/*     */     {
/*     */       
/* 144 */       return new ClusterIPResponse(AccountRegistry.APP().getIPPorts(PeerRegistry.SYSTEM().getIPPorts(childrenKeys)));
/*     */     }
/*     */ 
/*     */ 
/*     */     
/* 149 */     return new ClusterIPResponse();
/*     */   }
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */   
/* 156 */   public static MulticastChildrenResponse retrieveMulticastChildrenIPs(MulticastChildrenRequest req) { return new MulticastChildrenResponse(AccountRegistry.APP().getIPPorts(PeerRegistry.SYSTEM().getIPPorts())); }
/*     */ 
/*     */ 
/*     */ 
/*     */   
/* 161 */   public static PeerAddressResponse retrievePeerAddress(PeerAddressRequest req) { return new PeerAddressResponse(PeerRegistry.SYSTEM().getAddress(req.getPeerID())); }
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */   
/* 167 */   public static PeerDisableStateResponse retrievePeerDisableState(PeerDisableStateRequest req) { return new PeerDisableStateResponse(PeerRegistry.SYSTEM().isServerDisabled(req.getPeerID())); }
/*     */ 
/*     */ 
/*     */ 
/*     */   
/* 172 */   public static SearchBrokerResponse retrieveBroker(SearchBrokerRequest req) { return new SearchBrokerResponse(PeerRegistry.SYSTEM().getBroker()); }
/*     */ 
/*     */ 
/*     */ 
/*     */   
/* 177 */   public static AllRegisteredIPsResponse retrieveAllRegisteredIPs() { return new AllRegisteredIPsResponse(PeerRegistry.SYSTEM().getIPPorts()); }
/*     */ 
/*     */ 
/*     */ 
/*     */   
/* 182 */   public static AllPeerNamesResponse retrieveAllPeerNames() { return new AllPeerNamesResponse(PeerRegistry.SYSTEM().getNames()); }
/*     */ 
/*     */ 
/*     */ 
/*     */   
/*     */   public static IsRootOnlineResponse isRootOnline(IsRootOnlineRequest req) {
/* 188 */     Clusters.SYSTEM().joinCluster(req.getRootID(), req.getChildKey());
/* 189 */     IPAddress rootIP = PeerRegistry.SYSTEM().getAddress(req.getRootID());
/* 190 */     if (rootIP != UtilConfig.NO_IP_ADDRESS)
/*     */     {
/* 192 */       return new IsRootOnlineResponse(rootIP, true);
/*     */     }
/*     */ 
/*     */     
/* 196 */     return new IsRootOnlineResponse(rootIP, false);
/*     */   }
/*     */ 
/*     */ 
/*     */ 
/*     */   
/* 202 */   public static void leaveCluster(LeaveClusterNotification notification) { Clusters.SYSTEM().leaveCluster(notification.getRootKey(), notification.getChildKey()); }
/*     */ 
/*     */ 
/*     */   
/*     */   public static UnregisterPeerResponse unregisterPeer(UnregisterPeerRequest req) throws PeerKeyNotExistedInRegistryException {
/* 207 */     AccountRegistry.APP().remove(req.getPeerKey());
/* 208 */     PeerRegistry.SYSTEM().unregister(req.getPeerKey());
/* 209 */     return new UnregisterPeerResponse(true);
/*     */   }
/*     */ }
