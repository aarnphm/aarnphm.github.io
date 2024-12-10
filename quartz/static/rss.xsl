<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  <xsl:output method="html" version="1.0" encoding="utf-8" indent="yes" />
  <xsl:template match="/">
    <html xmlns="http://www.w3.org/1999/xhtml">
      <head>
        <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1" />
        <title>RSS - <xsl:value-of select="/rss/channel/title" /></title>
        <link rel="icon" href="./static/icon.webp" />
        <link href="/index.css" rel="stylesheet" type="text/css" />
        <style type="text/css">
          body{max-width:768px;margin:0 auto;font-size:16px;line-height:1.5em}section{margin:30px
          15px}h1{font-size:2em;margin:.67em
          0;line-height:1.125em}article{padding-bottom:1rem;border-bottom:1px solid
          var(--lightgray)}h2{margin-bottom:0}
        </style>
      </head>
      <body>
        <header class="rss">
          <hgroup>
            <h1 class="article-title">
              <xsl:value-of select="/rss/channel/title" />
            </h1>
            <p>
              <xsl:value-of select="/rss/channel/description" />
            </p>
          </hgroup>
          <a target="_blank">
            <xsl:attribute name="href">
              <xsl:value-of select="/rss/channel/link" />
            </xsl:attribute>main site
            &#x2192;</a>

          <p>There is also a <a href="/feed.xml">Atom feed</a> of the site. They are <a
              href="https://news.ycombinator.com/item?id=26168493" target="_blank">semantically
            different but achieve the same thing.</a></p>

          <blockquote class="callout tip" data-callout="tip">
            <div class="callout-title" dir="auto">
              <div class="callout-icon" dir="auto"></div>
              <div class="callout-title-inner" dir="auto">
                <p dir="auto">subscribe</p>
              </div>
            </div>
            <div class="callout-content" dir="auto">
              <p dir="auto">On most slash-command supported interface, you can use the following <code>/feed
                subscribe <xsl:value-of select="/rss/channel/link" />/index.xml</code></p>
            </div>
          </blockquote>
        </header>
        <hr />
        <main class="rss">
          <xsl:apply-templates select="/rss/channel/item" />
        </main>
      </body>
    </html>
  </xsl:template>

  <xsl:template match="item">
    <article xmlns="http://www.w3.org/1999/xhtml">
      <hgroup>
        <h2>
          <a>
            <xsl:attribute name="href">
              <xsl:value-of select="link" />
            </xsl:attribute>
            <xsl:value-of select="title" />
          </a>
        </h2>
        <p class="description">
          <xsl:value-of select="description" />
        </p>
      </hgroup>
      <menu class="tags">
        <xsl:for-each select="category">
          <li class="tag">
            <xsl:value-of select="." />
          </li>
        </xsl:for-each>
      </menu>
      <div class="published">
        <span lang="fr" class="metadata" dir="auto">dernière modification par <xsl:value-of
            select="pubDate" /></span>
      </div>
    </article>
  </xsl:template>
</xsl:stylesheet>
