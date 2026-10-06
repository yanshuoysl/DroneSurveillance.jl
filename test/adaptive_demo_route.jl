# Helpers for the visualization's route through known, static obstacle cells.
function demo_grid_path(pomdp, start, target)
    DS = DroneSurveillanceAdaptive
    free(pos) = 1 <= pos[1] <= pomdp.size[1] && 1 <= pos[2] <= pomdp.size[2] && pos ∉ pomdp.obstacles
    free(start) && free(target) || return nothing
    queue = [start]
    parents = Dict(start => start)
    next = 1
    while next <= length(queue)
        pos = queue[next]
        next += 1
        if pos == target
            path = [target]
            while path[end] != start
                push!(path, parents[path[end]])
            end
            return reverse(path)
        end
        for direction in (DS.DSPos(1, 0), DS.DSPos(0, 1), DS.DSPos(-1, 0), DS.DSPos(0, -1))
            neighbor = pos + direction
            if free(neighbor) && !haskey(parents, neighbor)
                parents[neighbor] = pos
                push!(queue, neighbor)
            end
        end
    end
    return nothing
end

function demo_waypoint_route(pomdp, waypoints)
    DS = DroneSurveillanceAdaptive
    position = pomdp.region_A
    heading = pomdp.initial_heading
    route = Int64[]
    for waypoint in waypoints
        path = demo_grid_path(pomdp, position, waypoint)
        path === nothing && return nothing
        for i in 2:length(path)
            direction = path[i] - path[i - 1]
            target_heading = 90 * (findfirst(==(direction), DS.HEADING_DIRS) - 1)
            turn = mod(target_heading - heading, 360)
            if turn == 270
                push!(route, DS.ACTIONS_DICT[:rotate_counterclockwise])
            else
                append!(route, fill(DS.ACTIONS_DICT[:rotate_clockwise], div(turn, 90)))
            end
            push!(route, DS.ACTIONS_DICT[:forward])
            heading = target_heading
        end
        position = waypoint
    end
    return route
end

function random_demo_environment(rng, waypoint)
    DS = DroneSurveillanceAdaptive
    grid_size = (5, 5)
    start, goal = DS.DSPos(1, 1), DS.DSPos(5, 5)
    # Keep the requested waypoint free and resample disconnected layouts.
    for _ in 1:100
        obstacles = DS.random_obstacles(grid_size; exclude=[start, waypoint, goal], rng=rng)
        pomdp = DS.DroneSurveillancePOMDP(size=grid_size, obstacles=obstacles)
        route = demo_waypoint_route(pomdp, [waypoint, goal])
        route === nothing || return pomdp, route
    end
    error("Could not generate a connected demonstration map")
end
