using Dates
using Images

Base.@kwdef mutable struct Html
    num_imgs::Int = 0
    body::String = ""
    head::String = "Output"
    timestamp::DateTime
    dir::String
end

function html_new()
    timestamp = Dates.now()
    dir = "$(Dates.format(timestamp, "yyyy-mm-dd__HH-MM-SS"))"
    mkpath("out/html/$dir/imgs")
    Html(timestamp=timestamp, dir=dir)
end

global curr_html::Html = html_new()

function html_get()
    curr_html
end

"""
Sets the global html, returning the old one
"""
function html_set(html::Html)
    global curr_html
    old = curr_html
    curr_html = html
    old
end

"""
Makes a fresh global html, returning the old one
"""
function html_fresh()
    html_set(html_new())
end

function html_body(body...)
    for b in body
        curr_html.body *= "\n\n" * string(b)
    end
end

# function add_style!(style)
#     curr_html.styles *= "\n\n" * string(style)
# end



html_img(img::Array{Float64,3}, attrs...; kwargs...) = html_img(Array(colorview(RGB,img)), attrs...; kwargs...)

function html_img(img::Matrix{RGB{Float64}}, attrs...; width="200px")
    curr_html.num_imgs += 1
    path = "imgs/img_$(curr_html.num_imgs).png"
    save("out/html/$(curr_html.dir)/$path", img)
    attrs = join(string.(attrs), " ")
    if !isnothing(width)
        attrs *= " width=$width"
    end
    res = if !isempty(attrs) "<img src=$(path) $attrs>" else "<img src=$(path)>" end
    res
end

html_gif(gif::Array{Float64,4}, attrs...; kwargs...) = html_gif(Array(colorview(RGB,gif)), attrs...; kwargs...)

function html_gif(gif::Array{RGB{Float64},3}, attrs...; fps=3, width="200px")

    # add progress bar along the bottom
    H,W,T = size(gif)
    bar_size = 4
    gif = cat(gif, fill(RGB(0.5,0.,0.), bar_size, W, T), dims=1)
    for t in 1:T
        # gif[1:bar_size,:,t] .= RGB(0.0,0.,0.)
        gif[end-bar_size+1:end, 1:round(Int, W*t/T), t] .= RGB(0.,0.75,0.)
    end

    curr_html.num_imgs += 1
    attrs = join(string.(attrs), " ")
    if !isnothing(width)
        attrs *= " width=$width"
    end

    attrs *= " class=\"anim\""

    # path = "imgs/img_$(curr_html.num_imgs).gif"
    # save("out/html/$(curr_html.dir)/$path", gif, fps=fps)
    # res = if !isempty(attrs) "<img src=$(path) $attrs>" else "<img src=$(path)>" end

    for t in 1:T
        path = "imgs/img_$(curr_html.num_imgs)___T$t.png"
        save("out/html/$(curr_html.dir)/$path", gif[:,:,t])
    end
    path = "imgs/img_$(curr_html.num_imgs)___T1.png"
    res = "<img src=\"$path\" $attrs >"
    res
end

function html_table(table::Matrix, attrs...)
    attrs = join(string.(attrs), " ")
    res = if isempty(attrs) "<table>" else "<table $attrs>" end
    for row in eachrow(table)
        res *= "\n\t<tr>"
        for x in row
            res *= "\n\t\t<td>$x</td>"
        end
        res *= "\n\t</tr>"
    end
    res *= "\n</table>"
    res
end

function html_tslider()
    html_body("""
    <div class="slidecontainer">
        <input type="range" class="slider_t"> [autoplay: <input type="checkbox" class="autoplayCheckbox">] t=<span class="show_t"></span>
    </div>
    """)
end


function html_render(;open=true, publish::Union{Bool,Nothing}=nothing, styles="css/styles.css", scripts="js/scripts.js")
    if isnothing(publish)
        publish = occursin("csail.mit.edu", gethostname()) && occursin("sketch", gethostname())
    end
    full_path = joinpath(Base.Filesystem.pwd(), "out/html", curr_html.dir)
    res = """
    <html>
        <head>
            <link rel="stylesheet" href="styles.css">
            <h1>$(curr_html.head) @ $(Dates.format(curr_html.timestamp, "yyyy-mm-dd HH:MM:SS"))</h1>
        </head>


        <body>

        <script src="scripts.js"></script>

        <div class="slidecontainer">
            <input type="range" class="slider_t"> [autoplay: <input type="checkbox" class="autoplayCheckbox">] t=<span class="show_t"></span>
        </div>

        <p> <b>Controls:</b> E: toggle autoplay; A/D: step time back/forward </p>


        $(curr_html.body)

        </body>

        <footer>
        <h4>Command to publish</h4>
        <code>julia> using Atari; Atari.html_publish("$full_path")</code>
        </footer>
    </html>
    """
    res = string(res)
    # println(res)
    write("out/html/$(curr_html.dir)/index.html",res)
    cp(styles, "out/html/$(curr_html.dir)/styles.css")
    cp(scripts, "out/html/$(curr_html.dir)/scripts.js")
    println("wrote to out/html/$(curr_html.dir)/index.html")
    open && !publish && open_in_default_browser("out/html/$(curr_html.dir)/index.html")
    publish && html_publish(full_path)
    
end

function html_publish(path)
    @assert isdir(path)
    while endswith(path,"/")
        path = path[:end-1] # we need to trim trailing "/" to make basename() work
    end
    publish_site = get_secret("publish_site") * "/" * basename(path)
    if occursin("csail.mit.edu", gethostname()) && occursin("sketch", gethostname())
        publish_dir = get_secret("publish_dir") * "/" * basename(path)
        println("On CSAIL network: copying files to public site")
        cp(path, publish_dir)
        println("See results at: $publish_site")
    else
        println("Not on CSAIL network: attempting rsync")
        publish_dir = get_secret("publish_dir")
        publish_ssh = get_secret("publish_ssh")
        Base.run(`rsync -avz $path $publish_ssh:$publish_dir/`)
        open_in_default_browser(publish_site)
    end
end

function get_secret(key)
    file = ".secret_$key"
    isfile(file) || error("file $file does not exist. If you'd like to use this feature, add the secret locally but do NOT commit it to git.")

    open(file) do f
        return read(f, String)
    end
end

function detectwsl()
    Sys.islinux() &&
    isfile("/proc/sys/kernel/osrelease") &&
    occursin(r"Microsoft|WSL"i, read("/proc/sys/kernel/osrelease", String))
end

function open_in_default_browser(url::AbstractString)::Bool
    try
        if Sys.isapple()
            Base.run(`open $url`)
            return true
        elseif Sys.iswindows() || detectwsl()
            # Base.run(`cmd.exe /s /c start "" /b $url`)
            url = "file://wsl.localhost/Ubuntu/home/jssteele/probabilistic-atari/$url"
            Base.run(`cmd.exe /s /c start chrome $url`)
            return true
        elseif Sys.islinux()'
            browser = "xdg-open"
            if isfile(browser)
                Base.run(`$browser $url`)
                return true
            else
                @warn "Unable to find `xdg-open`. Try `apt install xdg-open`"
                return false
            end
        else
            return false
        end
    catch ex
        return false
    end
end


const fresh = html_fresh
const render = html_render
