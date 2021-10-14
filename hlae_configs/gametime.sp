#include <sourcemod>
#include <sdktools>

public Plugin myinfo =
{
    name = "Durst Game Time Logger Plugin",
    author = "David Durst",
    description = "Log the current server time in the info every frame",
    version = "1.0",
    url = "https://davidbdurst.com/"
};

public OnGameFrame()
{
    int cur_tick = GetGameTickCount();
    SetHudTextParams(0.2, 0.2, 0.05, 0, 255, 0, 255, 0, 0.0, 0.0, 0.0);
    for(int i = 1; i < MaxClients; i++)
    {
        if(IsClientConnected(i) && IsClientInGame(i))
        {
            ShowHudText(i, 1, "Tick: %i", cur_tick);
        }
    }
}
