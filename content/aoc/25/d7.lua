---@alias Grid string[][]

---@param filename string Path to input file.
---@return Grid grid 2D grid
---@return integer start_col index for 'S'
local function parse(filename)
  local grid = {}
  local start_col = nil

  local file = io.open(filename, "r")
  if not file then
    error("Could not open file: " .. filename)
  end

  local row = 1
  for line in file:lines() do
    grid[row] = {}
    for col = 1, #line do
      local ch = line:sub(col, col)
      grid[row][col] = ch
      if ch == "S" then
        start_col = col
      end
    end
    row = row + 1
  end

  file:close()

  return grid, start_col
end

---@param grid Grid 2D grid
---@param start_col integer column index of 'S'
---@return integer splits number of times beams hit splitters
local function p1(grid, start_col)
  local splits = 0
  local rows = #grid
  local cols = grid[1] and #grid[1] or 0

  local beams = { [start_col] = true }

  for row = 2, rows do
    ---@type table<integer, boolean>
    local nextb = {}

    for col in pairs(beams) do
      if col >= 1 and col <= cols then
        local cell = grid[row][col]
        if cell == "^" then
          splits = splits + 1
          if col - 1 >= 1 then
            nextb[col - 1] = true
          end
          if col + 1 <= cols then
            nextb[col + 1] = true
          end
        elseif cell == "." then
          nextb[col] = true
        end
      end
    end

    beams = nextb
    if not next(beams) then
      break
    end
  end

  return splits
end

---@param grid Grid 2D grid
---@param start_col integer column index of 'S'
---@return integer total Total number of distinct timelines
local function p2(grid, start_col)
  local rows = #grid
  local cols = grid[1] and #grid[1] or 0

  local timelines = { [start_col] = 1 }

  for row = 2, rows do
    ---@type table<integer, boolean>
    local nextt = {}

    for col, count in pairs(timelines) do
      if col >= 1 and col <= cols then
        local cell = grid[row][col]

        if cell == "^" then
          if col - 1 >= 1 then
            nextt[col - 1] = (nextt[col - 1] or 0) + count
          end
          if col + 1 <= cols then
            nextt[col + 1] = (nextt[col + 1] or 0) + count
          end
        elseif cell == "." then
          nextt[col] = (nextt[col] or 0) + count
        end
      end
    end

    timelines = nextt
    if not next(timelines) then
      break
    end
  end

  local total = 0
  for _, count in pairs(timelines) do
    total = total + count
  end

  return total
end

local grid, start_col = parse("d7.txt")
print("p1: ", p1(grid, start_col))
print("p2: ", p2(grid, start_col))
